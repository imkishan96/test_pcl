// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
//#include "pch.h"


#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <examples/example.hpp> // Include short list of convenience functions for rendering

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
// extra includes
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
// new from segmentioan 
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <Eigen/Dense>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>




// Struct for managing rotation of pointcloud view
struct state {
    state() : yaw(0.0), pitch(0.0), last_x(0.0), last_y(0.0),
        ml(false), offset_x(0.0f), offset_y(0.0f) {}
    double yaw, pitch, last_x, last_y; bool ml; float offset_x, offset_y; 
};

using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

// Helper functions
void register_glfw_callbacks(window& app, state& app_state);
void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_ptr>& points);

pcl_ptr points_to_pcl(const rs2::points& points)
{
    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}

float3 colors[] { { 0.8f, 0.1f, 0.3f }, 
                  { 0.1f, 0.9f, 0.5f },
                };


void show_pcl_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.5);
    viewer.addPointCloud(cloud);

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
}


int main(int argc, char* argv[]) try
{   
    window app(1280, 720, "RealSense Pointcloud Example");
    state app_state;
    register_glfw_callbacks(app, app_state);

    rs2::pointcloud pc;
    rs2::points points;
    rs2::pipeline pipe;
    rs2::frame fff;
    rs2::frameset all_frames;
    rs2::config cfg;
    auto depth = fff.as<rs2::depth_frame>();
    auto motion_frame = fff.as<rs2::motion_frame>();
    pcl_ptr pcl_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl_ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass, pass_1;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg_n;
    pcl::IntegralImageNormalEstimation < pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_DEPTH, RS2_FORMAT_ANY);
    pipe.start(cfg);
    all_frames = pipe.wait_for_frames();
    motion_frame = all_frames.first(RS2_STREAM_ACCEL);
    depth = all_frames.get_depth_frame();
    pass.setInputCloud(pcl_points);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-1,1);
    pass_1.setInputCloud(cloud_filtered);
    pass_1.setFilterFieldName("z");
    pass_1.setFilterLimits(0, 2);
    //seg.setOptimizeCoefficients(true);
    //seg.setModelType(pcl::SACMODEL_PLANE);
    //seg.setMethodType(pcl::SAC_RANSAC);
    //
    //seg.setInputCloud(cloud_filtered);
    //seg.setProbability(0.99);
    //
    extract.setInputCloud(cloud_filtered);
    extract.setNegative(false);   
    ne.setInputCloud(cloud_filtered);
    //ne.setKSearch(5);
    //ne.setSearchMethod(tree);
    //ne.setRadiusSearch(0.05);
    ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.2f);
    ne.setNormalSmoothingSize(10.0f);
    ne.useSensorOriginAsViewPoint();

    seg_n.setOptimizeCoefficients(true);
    seg_n.setInputCloud(cloud_filtered);
    seg_n.setInputNormals(cloud_normals);
    seg_n.setModelType(pcl::SACMODEL_PLANE);
    seg_n.setMethodType(pcl::SAC_RANSAC);
    seg_n.setDistanceThreshold(0.05);
    seg_n.setMaxIterations(100);

    std::vector<pcl_ptr> layers;
    std::vector<float> PN, A;
    double dot, norm, theta;
    Eigen::Vector3f Axis_to_find;
    Eigen::VectorXf coefficients_n;


    while (app)
    {
        for (int i = 0; i <= 5; i++)
        {
            all_frames = pipe.wait_for_frames();
            motion_frame = all_frames.first(RS2_STREAM_ACCEL);
            depth = all_frames.get_depth_frame();
            cout << motion_frame.get_motion_data() << "motion data" << endl;
        }
        

        points = pc.calculate(depth);
        *pcl_points = *points_to_pcl(points);

        *cloud_filtered = *pcl_points;
        //pass.filter(*cloud_filtered);
        //pass_1.filter(*cloud_filtered);
          
        A.push_back(motion_frame.get_motion_data().x);
        A.push_back(motion_frame.get_motion_data().y);
        A.push_back(motion_frame.get_motion_data().z);
        //Axis_to_find.x() = A[0];
        //Axis_to_find.y() = A[1];
        //Axis_to_find.z() = A[2];
        //seg.setAxis(Axis_to_find);
        //seg.segment(*inliers, *coefficients);

        
        ne.compute(*cloud_normals);

        seg_n.segment(*inliers, *coefficients);

        //sac_model->setInputCloud = *cloud_filtered;

        //cout << *coefficients << endl;
        //cout << typeid(motion_frame.get_motion_data().x).name() << endl;

        /*
        PN.push_back(coefficients->values[0]);
        PN.push_back(coefficients->values[1]);
        PN.push_back(coefficients->values[2]);

        double dot = A[0] * PN[0] + A[1] * PN[1] + A[2] * PN[2];
        double norm = sqrt( A[0] * A[0] + A[1] * A[1] + A[2] * A[2] ) * sqrt( PN[0]* PN[0] + PN[1] * PN[1] + PN[2] * PN[2] );
        double theta = acos(dot / norm);

        cout << theta <<"= Theta (radian)" <<endl;
        cout << theta * 180 / PI << "Theta(degree)" << endl;

        
            if (inliers->indices.size() == 0)
        {
            PCL_ERROR("Could not estimate a planar model for the given dataset.");
            return (-1);
        }
        std::cerr << "Model coefficients: " << coefficients->values[0] << " "
            << coefficients->values[1] << " "
            << coefficients->values[2] << " "
            << coefficients->values[3] << std::endl;
        //std::cerr << "Model inliers: " << inliers->indices.size() << std::endl;

        */
        extract.setIndices(inliers);
        extract.filter(*cloud_p);
        cout << inliers << endl;
        

        layers.push_back(pcl_points);
        layers.push_back(cloud_p);
        draw_pointcloud(app, app_state, layers);
        layers.clear();
        //A.clear();
        //PN.clear();

    }

    //pipe.stop();
    return EXIT_SUCCESS;
}

catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

// Registers the state variable and callbacks to allow mouse control of the pointcloud
void register_glfw_callbacks(window& app, state& app_state)
{
    app.on_left_mouse = [&](bool pressed)
    {
        app_state.ml = pressed;
    };

    app.on_mouse_scroll = [&](double xoffset, double yoffset)
    {
        app_state.offset_x += static_cast<float>(xoffset);
        app_state.offset_y += static_cast<float>(yoffset);
    };

    app.on_mouse_move = [&](double x, double y)
    {
        if (app_state.ml)
        {
            app_state.yaw -= (x - app_state.last_x);
            app_state.yaw = std::max(app_state.yaw, -120.0);
            app_state.yaw = std::min(app_state.yaw, +120.0);
            app_state.pitch += (y - app_state.last_y);
            app_state.pitch = std::max(app_state.pitch, -80.0);
            app_state.pitch = std::min(app_state.pitch, +80.0);
        }
        app_state.last_x = x;
        app_state.last_y = y;
    };

    app.on_key_release = [&](int key)
    {
        if (key == 32) // Escape
        {
            app_state.yaw = app_state.pitch = 0; app_state.offset_x = app_state.offset_y = 0.0;
        }
    };
}

// Handles all the OpenGL calls needed to display the point cloud
void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_ptr>& points)
{
    // OpenGL commands that prep screen for the pointcloud
    glPopMatrix();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    float width = app.width(), height = app.height();

    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(60, width / height, 0.01f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

    glTranslatef(0, 0, +0.5f + app_state.offset_y * 0.05f);
    glRotated(app_state.pitch, 1, 0, 0);
    glRotated(app_state.yaw, 0, 1, 0);
    glTranslatef(0, 0, -0.5f);

    glPointSize(width / 640);
    glEnable(GL_TEXTURE_2D);

    int color = 0;

    for (auto&& pc : points)
    {
        auto c = colors[(color++) % (sizeof(colors) / sizeof(float3))];

        glBegin(GL_POINTS);
        glColor3f(c.x, c.y, c.z);

        /* this segment actually prints the pointcloud */
        for (int i = 0; i < pc->points.size(); i++)
        {
            auto&& p = pc->points[i];
            if (p.z)
            {
                // upload the point and texture coordinates only for points we have depth data for
                glVertex3f(p.x, p.y, p.z);
            }
        }

        glEnd();
    }

    // OpenGL cleanup
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
    glPushMatrix();
}