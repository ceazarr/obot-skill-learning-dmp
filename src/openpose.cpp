#include <openpose/headers.hpp>
#include <openpose/flags.hpp>
#include <algorithm>
// OpenCV Headers
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/opencv.hpp>

// RealSense Headers
#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_frame.hpp>

#include <vector>
#include <array>
#include <Eigen/Dense>
#include <fstream>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/types.h>
//Visualization
#include "matplotlibcpp.h"
// own utils:
#include <Visualizer.h>
#include <AprilTagDetector.h>

// USING PCL
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// Threading
#include <thread>
#include <mutex>
#include <chrono>

#include "Visualizer.h"

#include <nlohmann/json.hpp>
#include <cmath>
#include <unordered_map>
#include <deque>
#include <limits>
#include <filesystem>


namespace fs = std::filesystem;
namespace plt = matplotlibcpp;

std::vector<std::pair<std::string, cv::Point2f> > extractKeypoints(
    const std::shared_ptr<std::vector<std::shared_ptr<op::Datum> > > &datumsPtr) {
    const std::vector<std::string> keypointName = {
        "Head", "Neck", "Right_Shoulder", "Right_Elbow", "Right_Wrist", "Left_Shoulder", "Left_Elbow", "Left_Wrist",
        "Hip"
    };

    std::vector<std::pair<std::string, cv::Point2f> > keypointswithNames; //save the result

    if (datumsPtr != nullptr && !datumsPtr->empty()) {
        const auto &poseKeypoints = datumsPtr->at(0)->poseKeypoints;

        for (auto person = 0; person < poseKeypoints.getSize(0); person++) {
            for (auto bodyPart = 0; bodyPart < keypointName.size(); bodyPart++) {
                const auto x = poseKeypoints[{person, bodyPart, 0}];
                const auto y = poseKeypoints[{person, bodyPart, 1}];
                const auto confidence = poseKeypoints[{person, bodyPart, 2}];
                if (confidence > 0.5) {
                    keypointswithNames.emplace_back(keypointName[bodyPart], cv::Point2f(x, y));
                } else {
                    keypointswithNames.emplace_back(keypointName[bodyPart], cv::Point2f(x, y));
                }
            }
        }
    }
    return keypointswithNames;
}

float getMedianDepth(const cv::Mat& depthMat, int x, int y, int window_size = 3)
{
    std::vector<uint16_t> valid_depth;
    const int half_window = window_size / 2;

    for (int dy = -half_window; dy <= half_window; ++dy) {
        for (int dx = -half_window; dx <= half_window; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < depthMat.cols && ny >= 0 && ny < depthMat.rows) {
                uint16_t raw_depth = depthMat.at<uint16_t>(ny, nx);

                if (raw_depth != 0) valid_depth.push_back(raw_depth);
            }
        }
    }
    if (valid_depth.empty()) {
        return 0.0f;
    }
    auto m = valid_depth.begin() + valid_depth.size() / 2;
    std::nth_element(valid_depth.begin(), m, valid_depth.end());
    return valid_depth[valid_depth.size() / 2] * 0.001f; //mm to meters
}

struct HandPoint {
    cv::Point2f pt{NAN, NAN};
    float conf{0.f};
    bool inferred{false};
    const char* source{"none"};
};

static std::unordered_map<std::string, HandPoint> lastHandPts;
static std::unordered_map<std::string, float> lastUpdateTs;

inline bool getKP(const op::Array<float>& rh, int person, int id, cv::Point2f& p, float& conf) {
    if (rh.getSize(0) <= person) return false;
    float x = rh[{person, id, 0}];
    float y = rh[{person, id, 1}];
    float c = rh[{person, id, 2}];
    p = {x, y}; conf = c;
    return conf > 0.f && std::isfinite(x) && std::isfinite(y);
}

inline cv::Point2f extrapolate(const cv::Point2f& a, const cv::Point2f& b, float r) {
    return cv::Point2f(a.x + r*(a.x - b.x), a.y + r*(a.y - b.y));
}

inline float handScale(const op::Array<float>& rh, int person){
    cv::Point2f p5, p0; float c5=0, c0=0;
    if (!getKP(rh, person, 5, p5, c5) || !getKP(rh, person, 0, p0, c0))
        return 40.f; // fallback px
    float d = static_cast<float>(cv::norm(p5 - p0));
    return std::max(20.0f, d);
}
std::vector<std::pair<std::string, cv::Point2f>> extractRightHandTips(
    const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr,
    float tauHigh = 0.35f, float tauLow = 0.25f, float emaAlpha = 0.35f, float holdSec = 0.5f,
    float distalRatio = 0.65f)
{
    std::vector<std::pair<std::string, cv::Point2f>> out;
    if (!datumsPtr || datumsPtr->empty()) return out;

    const auto& hands = datumsPtr->at(0)->handKeypoints;
    if (hands.size() < 2) return out;
    const auto& rh = hands[1];
    if (rh.getSize(0) <= 0) return out;
    const int person = 0;

    static auto t0 = std::chrono::steady_clock::now();
    float now = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();

    auto commit = [&](const std::string& name, HandPoint hp){
        // EMA smoothing if we have a previous value
        auto it = lastHandPts.find(name);
        if (it != lastHandPts.end() && std::isfinite(it->second.pt.x) && std::isfinite(hp.pt.x)) {
            hp.pt = (1.0f - emaAlpha)*it->second.pt + emaAlpha*hp.pt;
            hp.conf = std::max(hp.conf, it->second.conf*0.2f);
        }
        lastHandPts[name] = hp;
        lastUpdateTs[name] = now;
        out.emplace_back(name, hp.pt);
    };

    auto hold_or_skip = [&](const std::string& name){
        auto itT = lastUpdateTs.find(name);
        auto itP = lastHandPts.find(name);
        if (itT != lastUpdateTs.end() && (now - itT->second) <= holdSec) {
            out.emplace_back(name, itP->second.pt); // hold last value
        }
    };

    auto robustOne = [&](const std::string& name,
                         int tip, int dip, int pip, int mcp) {
        //  raw tip with hysteresis
        cv::Point2f pTip; float cTip=0;
        bool hasTip = getKP(rh, person, tip, pTip, cTip);
        bool acceptTip = false;

        auto itPrev = lastHandPts.find(name);
        float prevConf = (itPrev==lastHandPts.end()? 0.f : itPrev->second.conf);

        if (hasTip && cTip >= tauHigh) acceptTip = true;
        else if (hasTip && cTip >= tauLow && prevConf >= tauHigh) acceptTip = true; // hysteresis

        if (acceptTip) {
            commit(name, HandPoint{pTip, cTip, false, "raw"});
            return;
        }

        // nearest upstream joint available (DIP -> PIP -> MCP)
        cv::Point2f pDip, pPip, pMcp; float cDip=0, cPip=0, cMcp=0;
        bool hasDip = getKP(rh, person, dip, pDip, cDip);
        bool hasPip = getKP(rh, person, pip, pPip, cPip);
        bool hasMcp = getKP(rh, person, mcp, pMcp, cMcp);

        if (hasDip && cDip >= tauHigh) {
            commit(name, HandPoint{pDip, cDip*0.8f, true, "neighbor"});
            return;
        }
        if (hasPip && cPip >= tauHigh) {
            commit(name, HandPoint{pPip, cPip*0.7f, true, "neighbor"});
            return;
        }
        if (hasMcp && cMcp >= tauHigh) {
            commit(name, HandPoint{pMcp, cMcp*0.6f, true, "neighbor"});
            return;
        }

        // Geometric extrapolation if we have two in a row
        if (hasDip && hasPip) {
            // tip ≈ DIP + r*(DIP - PIP) (limit length by hand scale)
            float scale = handScale(rh, person);
            cv::Point2f guess = extrapolate(pDip, pPip, distalRatio);
            if (cv::norm(guess - pDip) > 0.9f*scale) {
                cv::Point2f dir = pDip - pPip;
                float len = std::max<float>(1e-3f, cv::norm(dir));
                dir *= (0.9f*scale/len);
                guess = pDip + dir;
            }
            commit(name, HandPoint{guess, std::max(cDip, cPip)*0.5f, true, "extrap"});
            return;
        }
        if (hasPip && hasMcp) {
            float scale = handScale(rh, person);
            cv::Point2f guess = extrapolate(pPip, pMcp, distalRatio*0.9f);
            if (cv::norm(guess - pPip) > 0.9f*scale) {
                cv::Point2f dir = pPip - pMcp;
                float len = std::max<float>(1e-3f, cv::norm(dir));
                dir *= (0.9f*scale/len);
                guess = pPip + dir;
            }
            commit(name, HandPoint{guess, std::max(cPip, cMcp)*0.45f, true, "extrap"});
            return;
        }

        // Hold last known for a short time
        hold_or_skip(name);
    };

    robustOne("Right_Thumb_Tip", /*tip*/4, /*dip*/3, /*pip*/2, /*mcp*/1);
    robustOne("Right_Index_Tip", /*tip*/8, /*dip*/7, /*pip*/6, /*mcp*/5);

    return out;
}

int runOpenPose() {

    try {
        // Set session and file paths
        std::string session_name = "3DTest";
        std::string base_path = "../data/recordings/";
        std::string outputVideoPath = base_path + session_name + "_skeleton_with.avi";
        std::string outputVideoPathWithTarget = base_path + session_name + "_skeleton_with_target_points.avi";

        int frameCount = 0;

        std::map<std::string, Eigen::Vector3d> Points3D;
        int outputInterval = 1; //Frequency
        auto lastPlotTime = std::chrono::steady_clock::now();
        auto prevTime = std::chrono::steady_clock::now();

        // Create a realsense pipeline
        rs2::pipeline pipe;
        rs2::config cfg;

        // Enable RGB and Depth streams
        int framerate = 30;
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, framerate);
        cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, framerate);

        pipe.start(cfg); // Start streaming
        rs2::align align_to_color(RS2_STREAM_COLOR);

        // Get depth stream profile
        rs2::pipeline_profile profile = pipe.get_active_profile();
        //rs2::stream_profile sp = profile.get_stream(RS2_STREAM_DEPTH);
        //auto depth_stream = sp.as<rs2::video_stream_profile>();
        //rs2_intrinsics intrinsics = depth_stream.get_intrinsics();
        auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        rs2_intrinsics color_intr = color_stream.get_intrinsics();



        // output video (to save video)
        cv::VideoWriter RGBwriter(outputVideoPathWithTarget, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), framerate,
                                  cv::Size(color_intr.width, color_intr.height));

        //output video with target points (to save video including the openpose output)
        cv::VideoWriter skeletonwriterwithRGB(outputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), framerate,
                                              cv::Size(color_intr.width, color_intr.height));

        //check if both videos are open
        if (!RGBwriter.isOpened() || !skeletonwriterwithRGB.isOpened()) {
            std::cerr << "Failed to open files!" << std::endl;
            return -1;
        }

        // Configure OpenPose
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};

        // Configure model folder
        op::WrapperStructPose poseConfig;
        poseConfig.modelFolder = "../models/";
        poseConfig.renderMode = op::RenderMode::Gpu;
        poseConfig.blendOriginalFrame = false;
        poseConfig.alphaKeypoint = 1.0f;
        poseConfig.alphaHeatMap = 0.0f;
        opWrapper.configure(poseConfig);

        op::WrapperStructHand poseHand;
        poseHand.enable = true;
        poseHand.detector = op::Detector::Body;
        poseHand.renderMode = op::RenderMode::Gpu;
        poseHand.netInputSize = {256, 256};
        //poseHand.netInputSize = {512, 512};
        poseHand.scalesNumber = 2;
        poseHand.scaleRange = 0.35f;
        poseHand.alphaKeypoint = 1.0f;
        poseHand.alphaHeatMap = 0.0f;
        opWrapper.configure(poseHand);

        opWrapper.start();


        bool running = true;

        // AprilTagDetector:
        std::shared_ptr<ObjectMap> objectMap = nullptr;
        AprilTagDetector ObjectDetector;


        nlohmann::json keypoints3d = nlohmann::json::array();
        int frameidx = 0;

        // ---------------------------- drawer related skill ---------------------------

        bool have_drawer0 = false;
        Eigen::Matrix4d T_cd0 = Eigen::Matrix4d::Identity();
        float t_drawer0 = -1.f;

        // -----------------------------------------------------------------------------

        const int DELAY_SECONDS = 2;
        const int BUFFER_SECONDS = 2;
        bool recording = false;
        auto program_start_time = std::chrono::steady_clock::now();
        std::deque<nlohmann::json> keypoint_buffer;


        while (running) {
            Points3D.clear();

            // --- Acquire & align frames
            rs2::frameset frames;
            try {
                frames = pipe.wait_for_frames(5000);  // 5s timeout
            } catch (const rs2::error& e) {
                std::cerr << "RealSense error: " << e.what() << std::endl;
                continue;
            }
            rs2::frameset aligned     = align_to_color.process(frames);
            rs2::video_frame color_frame = aligned.get_color_frame();
            rs2::depth_frame depth_frame = aligned.get_depth_frame();

            // --- Use *aligned* depth intrinsics for everything in this frame
            //auto dprof = depth_frame.get_profile().as<rs2::video_stream_profile>();
            //rs2_intrinsics intr = dprof.get_intrinsics();

            // --- Make cv::Mat
            cv::Mat color(
                cv::Size(color_frame.get_width(), color_frame.get_height()),
                CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat rgb; cv::cvtColor(color, rgb, cv::COLOR_BGR2RGB);
            cv::Mat depthMat(
                cv::Size(depth_frame.get_width(), depth_frame.get_height()),
                CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

            // --- OpenPose inference
            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(rgb);
            auto datumsPtr = opWrapper.emplaceAndPop(imageToProcess);
            if (!datumsPtr || datumsPtr->empty()) {
                std::cerr << "OpenPose failed to process frame\n";
                RGBwriter.write(color);
                cv::imshow("RGB + OpenPose Detection", color);
                if (cv::waitKey(1) == 27) running = false;
                continue;
            }
            if (objectMap == nullptr) {
                objectMap = std::move(ObjectDetector.DetectObjects(color, color_intr)); // use COLOR intrinsics
            } else {
                *objectMap = *ObjectDetector.DetectObjects(color, color_intr);
            }

            // --- Compose skeleton overlay
            cv::Mat skeletonMat;
            cv::cvtColor(OP_OP2CVMAT(datumsPtr->at(0)->cvOutputData), skeletonMat, cv::COLOR_RGB2BGR);
            cv::Mat skeletonWithRGB = color.clone();
            cv::Mat grayskeleton, mask;
            cv::cvtColor(skeletonMat, grayskeleton, cv::COLOR_BGR2GRAY);
            cv::threshold(grayskeleton, mask, 1, 255, cv::THRESH_BINARY);
            skeletonMat.copyTo(skeletonWithRGB, mask);
            if (objectMap) {
                for (const auto& kv : *objectMap) {
                    const std::string& name = kv.first;   // e.g., "PlasticCupInstance1"
                    const Object& obj = kv.second;        // has .pose as 4x4 (camera->object)

                    // Project the object center to pixels
                    float P[3] = { (float)obj.pose(0,3), (float)obj.pose(1,3), (float)obj.pose(2,3) };
                    float px[2];
                    rs2_project_point_to_pixel(px, &color_intr, P);

                    // Use tag id if your Object exposes it; otherwise fall back to the instance name
                    std::string label = name; // or std::to_string(obj.tag_id) if available
                    cv::putText(skeletonWithRGB, label,
                                cv::Point((int)std::round(px[0]), (int)std::round(px[1])),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);
                }
            }

            // --- AprilTag detection with current intrinsics
            //if (!objectMap) objectMap = std::move(ObjectDetector.DetectObjects(color, color_intr));
            //else            *objectMap = *ObjectDetector.DetectObjects(color, color_intr);

            // --- 2D → 3D for BODY keypoints (camera frame)
            auto keypoints = extractKeypoints(datumsPtr);
            for (const auto& kp : keypoints) {
                const auto& name = kp.first;
                int x = (int)kp.second.x, y = (int)kp.second.y;
                if (0 <= x && x < color_frame.get_width() && 0 <= y && y < color_frame.get_height()) {
                    float d = depthMat.at<uint16_t>(y, x) * 0.001f;
                    if (d <= 0.1f || d > 10.0f || std::isnan(d)) {
                        d = getMedianDepth(depthMat, x, y);
                        if (d <= 0.1f) continue;
                    }
                    float P[3], px[2] = { kp.second.x, kp.second.y };
                    rs2_deproject_pixel_to_point(P, &color_intr, px, d);
                    Points3D[name] = Eigen::Vector3d(P[0], P[1], P[2]);
                }
            }

            // --- 2D → 3D for HAND TIPS (camera frame)
            auto handTips = extractRightHandTips(datumsPtr, 0.35f, 0.25f, 0.35f, 0.5f, 0.65f);
            for (const auto& tip : handTips) {
                int x = (int)tip.second.x, y = (int)tip.second.y;
                if (0 <= x && x < color_frame.get_width() && 0 <= y && y < color_frame.get_height()) {
                    float d = depthMat.at<uint16_t>(y, x) * 0.001f;
                    if (d <= 0.1f || d > 10.0f || std::isnan(d)) {
                        d = getMedianDepth(depthMat, x, y);
                        if (d <= 0.1f) continue;
                    }
                    float P[3], px[2] = { tip.second.x, tip.second.y };
                    rs2_deproject_pixel_to_point(P, &color_intr, px, d);
                    Points3D[tip.first] = Eigen::Vector3d(P[0], P[1], P[2]);
                }
            }

            // --- Current object poses
            bool have_table = (objectMap && objectMap->count("table1"));
            bool have_cup   = (objectMap && objectMap->count("PlasticCupInstance1"));
            bool have_drawer = (objectMap && objectMap->count("drawer"));

            Eigen::Matrix4d T_ct = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d T_co = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d T_cd = Eigen::Matrix4d::Identity();

            if (have_table) T_ct = (*objectMap)["table1"].pose;               // table in camera
            if (have_cup)   T_co = (*objectMap)["PlasticCupInstance1"].pose;  // cup   in camera
            if (have_drawer) T_cd = (*objectMap)["drawer"].pose;              // drawer in camera
//
            if (have_drawer && !have_drawer0) {
                T_cd0 = T_cd;
                have_drawer0 = true;
                t_drawer0 = std::chrono::duration<float>(std::chrono::steady_clock::now() - program_start_time).count();
            }
            Eigen::Matrix4d T_tc = T_ct.inverse();
            //Eigen::Matrix4d T_dc0 = T_cd0.inverse();
            Eigen::Matrix4d T_dc0 = Eigen::Matrix4d::Identity(); // T_drawer0_camera
            if (have_drawer0) {
                T_dc0 = T_cd0.inverse();
             }

            // --- Start recording after delay
            auto now = std::chrono::steady_clock::now();
            float seconds_since_start = std::chrono::duration<float>(now - program_start_time).count();
            if (seconds_since_start > DELAY_SECONDS) recording = true;

            if (recording) {
                nlohmann::json j;
                j["frame"]        = frameidx++;
                j["have_table"]   = have_table;
                j["have_cup"]     = have_cup;
                j["have_drawer"]  = have_drawer;
                j["table_id"]     = "table1";
                j["cup_id"]       = "PlasticCupInstance1";
                j["drawer_id"]    = "drawer";
                j["frame_source"] = "camera";

                // Row-major helper
                auto matToRowMajor = [](const Eigen::Matrix4d& M){
                    Eigen::Matrix<double,4,4,Eigen::RowMajor> R = M;
                    return std::vector<double>(R.data(), R.data()+16);
                };

                if (have_drawer)  j["T_cd"]  = matToRowMajor(T_cd);   // camera -> drawer (current)
                if (have_drawer0) j["T_cd0"] = matToRowMajor(T_cd0);  // camera -> drawer at first detection
                if (have_drawer0) {
                     j["T_cd0"] = matToRowMajor(T_cd0);  // The static reference pose (camera -> drawer0)
                     j["T_d0_c"] = matToRowMajor(T_dc0); // The static transform (drawer0 -> camera)
                     // the current drawer pose relative to the start pose
                     Eigen::Matrix4d T_d0_d = T_dc0 * T_cd;
                     j["T_d0_d"] = matToRowMajor(T_d0_d); // Pose of current drawer in drawer0's frame
//
                }

                // Transforms
                j["T_ct"] = matToRowMajor(T_ct);
                j["T_co"] = matToRowMajor(T_co);

                // pts_cam (always in camera frame)
                nlohmann::json pts_cam_json = nlohmann::json::object();
                const std::vector<std::string> cam_names = {
                    "Right_Wrist","Right_Index_Tip","Right_Thumb_Tip",
                    "Neck","Right_Shoulder","Right_Elbow","Hip"
                };
                for (const auto& nm : cam_names) {
                    auto itp = Points3D.find(nm);
                    if (itp != Points3D.end()) {
                        const auto& p = itp->second;
                        pts_cam_json[nm] = {p.x(), p.y(), p.z()};
                    }
                }
                j["pts_cam"] = pts_cam_json;
                if (have_drawer0) {
                    auto camToDrawer0 = [&](const Eigen::Vector3d& p_cam){
                        Eigen::Vector4d h; h << p_cam, 1.0;
                        Eigen::Vector4d q = T_dc0 * h;
                        return q.head<3>();
                    };
                    nlohmann::json drawer0_frame = nlohmann::json::object();
                    const std::vector<std::string> joints_tbl = {
                        "Neck","Right_Shoulder","Right_Elbow","Right_Wrist","Hip",
                        "Right_Index_Tip","Right_Thumb_Tip"
                    };
                    for (const auto& nm : joints_tbl) {
                        auto itp = Points3D.find(nm);
                        if (itp != Points3D.end()) {
                            Eigen::Vector3d  p_d0= camToDrawer0(itp->second);
                            drawer0_frame[nm] = { p_d0.x(), p_d0.y(), p_d0.z() };
                        }
                    }
                    j["drawer0_frame"] = drawer0_frame;
                }

                // Optional: table-frame for backward compatibility (only if table is available)
                if (have_table) {
                    auto camToTable = [&](const Eigen::Vector3d& p_cam){
                        Eigen::Vector4d h; h << p_cam, 1.0;
                        Eigen::Vector4d q = T_tc * h;
                        return q.head<3>();
                    };
                    nlohmann::json table_frame = nlohmann::json::object();
                    const std::vector<std::string> joints_tbl = {
                        "Neck","Right_Shoulder","Right_Elbow","Right_Wrist","Hip",
                        "Right_Index_Tip","Right_Thumb_Tip"
                    };
                    for (const auto& nm : joints_tbl) {
                        auto itp = Points3D.find(nm);
                        if (itp != Points3D.end()) {
                            Eigen::Vector3d p_tab = camToTable(itp->second);
                            table_frame[nm] = { p_tab.x(), p_tab.y(), p_tab.z() };
                        }
                    }
                    j["table_frame"] = table_frame;
                }

                // Grip proxies
                double pinch = -1.0, w2cup = -1.0, w2drawer = -1.0;
                auto itI = Points3D.find("Right_Index_Tip");
                auto itT = Points3D.find("Right_Thumb_Tip");
                if (itI != Points3D.end() && itT != Points3D.end())
                    pinch = (itI->second - itT->second).norm();
                //auto itW = Points3D.find("Right_Wrist");
                //if (itW != Points3D.end() && have_cup) {
                //    Eigen::Vector3d cup_c = T_co.block<3,1>(0,3); //----------------- uncomment for non-drawer skill ----------------------//
                //    w2cup = (itW->second - cup_c).norm();
                //}
                j["pinch_m"]        = pinch;
                j["wrist_to_cup_m"] = w2cup;
                auto itW = Points3D.find("Right_Wrist");
                if (itW != Points3D.end() && have_drawer) {
                    Eigen::Vector3d drawer_c = T_cd.block<3,1>(0,3); // current drawer center in camera frame
                    w2drawer = (itW->second - drawer_c).norm();
                }
                j["wrist_to_drawer_m"] = w2drawer;
//
                // Timestamps
                auto now2 = std::chrono::steady_clock::now();
                j["timestamp"]  = std::chrono::duration<float>(now2 - program_start_time).count();
                j["t_hw_ms"]    = color_frame.get_timestamp(); // RealSense hardware time (ms)
                j["ts_domain"]  = (int)color_frame.get_frame_timestamp_domain();

                //if (have_cup) {
                //    const Object& cup = objectMap->at("PlasticCupInstance1");
                //    Eigen::Vector3d cup_cam = cup.pose.block<3,1>(0,3);
                //    j["cup_pose_cam"] = { cup_cam.x(), cup_cam.y(), cup_cam.z() };
//
                //    if (Points3D.find("Right_Wrist") != Points3D.end()) {
                //        double d = (Points3D["Right_Wrist"] - cup_cam).norm();
                //        j["wrist_to_cup_m"] = d;
                //    } else {
                //        j["wrist_to_cup_m"] = -1.0;
                //    }
                //} else {
                //    j["wrist_to_cup_m"] = -1.0;
                //}

                keypoint_buffer.push_back(j);
            }

            // --- Write videos & UI
            RGBwriter.write(color);
            skeletonwriterwithRGB.write(skeletonWithRGB);
            cv::imshow("RGB + OpenPose Detection", skeletonWithRGB);

            // --- Exit & save
            int key = cv::waitKey(1);
            if (key == 27) { // ESC
                running = false;

                // Trim last BUFFER_SECONDS using timestamps only
                if (!keypoint_buffer.empty()) {
                    float last_ts = keypoint_buffer.back()["timestamp"];
                    float cutoff = last_ts - BUFFER_SECONDS;
                    while (!keypoint_buffer.empty() && keypoint_buffer.back()["timestamp"].get<float>() > cutoff) {
                        keypoint_buffer.pop_back();
                    }
                }

                // Save to file
                nlohmann::json out = nlohmann::json::array();
                for (auto& jf : keypoint_buffer) out.push_back(jf);
                const std::string fn = base_path + session_name + "_keypoints3d_drawerWithRetreating22.json";
                std::ofstream ofs(fn);
                if (!ofs) {
                    std::cerr << "Failed to open " << fn << " for writing.\n";
                } else {
                    ofs << out.dump(4);
                    std::cout << " ---> wrote " << out.size() << " keyframes\n";
                }
            }
        }

        nlohmann::json json_dummy;
        json_dummy["DMP_name"] = "dummy_dmp";
        //AndreiUtils::writeJsonFile("../data/DMP_dummy.json",json_dummy);

        skeletonwriterwithRGB.release();
        RGBwriter.release();

        std::cout << "The Process is finished" << std::endl;

        return 0;
    } catch (const std::exception &) {
        return -1;
    }
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    runOpenPose();
}