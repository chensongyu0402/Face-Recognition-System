#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <ctime>
#include <termios.h>
#include <poll.h>
#include <string>
#include <ctime>
#include <unistd.h> 

using namespace cv;
using namespace std;

// 1. define cascadeClassfier
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
clock_t   now;
Ptr<face::FaceRecognizer> model;
char img_name[15];
int img_cnt = 0;

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
    uint32_t yres_virtual;
};

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path );
int display_image(cv::Mat& img, std::ofstream& fb, struct framebuffer_info& fb_info);
Mat detect( Mat frame );

int main ( int argc, const char *argv[] )
{
    model = face::LBPHFaceRecognizer::create();
    model->read("./weight.xml");

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };


    // variable to store the frame get from video stream
    cv::Mat frame;
    cv::Size2f frame_size;
    
    
    struct termios t;
    struct pollfd fds[1] = {{fd:0, events:POLLIN, 0}};
    char img_name[10];
    int count = 1;
    int flag = 0;

    // open video stream device
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77bbebdcbfb341e3d4355c1
    cv::VideoCapture camera(2);
    //camera.set(CV_CAP_PROP_FRAME_WIDTH, 750);
    //camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
 
    int frame_width = camera.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = camera.get(CV_CAP_PROP_FRAME_HEIGHT);
    double frame_fps = camera.get(CV_CAP_PROP_FPS);
    std::cout << frame_fps << std::endl;

    // Turn off stdin enter
    tcgetattr(0, &t);
    t.c_lflag &= ~ICANON;
    tcsetattr(0, TCSANOW, &t);
    

    // get info of the framebuffer
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");

    camera.set(cv::CAP_PROP_FRAME_WIDTH, 400);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 300);
    camera.set(CV_CAP_PROP_BUFFERSIZE, 1);

    // open the framebuffer device
    // http://www.cplusplus.com/reference/fstream/ofstream/ofstream/
    // ofs = ......
    //cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(800,600));

    // check if video stream device is opened success or not
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a9d2ca36789e7fcfe7a7be3b328038585
    if( !camera.isOpened() )
    {
        std::cerr << "Could not open video device." << std::endl;
        return 1;
    }


    // set propety of the frame
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a8c6d8c2d37505b5ca61ffd4bb54e9a7c
    // https://docs.opencv.org/3.4.7/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    int  img_cnt = 0;
    char c;
    while ( true)
    {
        // get video frame from stream
        // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
        // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a199844fb74226a28b3ce3a39d1ff6765
        bool ret = camera.read(frame);
        if (!ret) {
            std::cerr << "Could not read video device." << std::endl;
        }
	//video.write(frame); 
	//printf("image size = %d x %d x %d\n", frame.cols, frame.rows, frame.channels());
        
        
        // transfer color space from BGR to BGR565 (16-bit image) to fit the requirement of the LCD
        // https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
        // https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0
	
        ret = poll(fds, 1, 10);
        if(ret == 1){
            c = std::cin.get();
            if (c == 'c'){
                now = clock();
                flag = 1;
            }
	    else if(c=='q'){
	    	break;
	    }
        }
        

        frame = detect(frame);

	if (flag == 1){
		cout << (clock() - now)/1000 << endl;
		flag = 0;
	}
		

	display_image(frame, ofs, fb_info);

	// get size of the video frame
        // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a146f8e8dda07d1365a575ab83d9828d1
        /*frame_size = frame.size();


	cv::cvtColor(frame, frame, cv::COLOR_BGR2BGR565);
        
        // output the video frame to framebufer row by row
        for ( int y = 0; y < frame_size.height; y++ )
        {
            // move to the next written position of output device framebuffer by "std::ostream::seekp()"
            // http://www.cplusplus.com/reference/ostream/ostream/seekp/
            ofs.seekp(y * fb_info.xres_virtual * 2);
            // write to the framebuffer by "std::ostream::write()"
            // you could use "cv::Mat::ptr()" to get the pointer of the corresponding row.
            // you also need to cacluate how many bytes required to write to the buffer
            // http://www.cplusplus.com/reference/ostream/ostream/write/
            // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a13acd320291229615ef15f96ff1ff738
            ofs.write(reinterpret_cast<char*>(frame.ptr(y)),frame_size.width*2);
        }*/

    }

    // closing video stream
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#afb4ab689e553ba2c8f0fec41b9344ae6
    camera.release();
    //video.release();
    return 0;
}

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path )
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.

    // open deive with linux system call "open( )"
    // https://man7.org/linux/man-pages/man2/open.2.html
    int fd = open(framebuffer_device_path, O_RDWR);
    // get attributes of the framebuffer device thorugh linux system call "ioctl()"
    // the command you would need is "FBIOGET_VSCREENINFO"
    // https://man7.org/linux/man-pages/man2/ioctl.2.html
    // https://www.kernel.org/doc/Documentation/fb/api.txt
    int attr = ioctl(fd, FBIOGET_VSCREENINFO, &screen_info);
    // put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;
    fb_info.yres_virtual = screen_info.yres_virtual;

    return fb_info;
};


int display_image(cv::Mat& img, std::ofstream& fb, struct framebuffer_info& fb_info){
    int x_offset=0; // value to make image to the middle
    cv::Size2i img_size;

    img_size.height = 600;
    img_size.width = 800;
    

    
    // Resize the image to fit screen
    cv::resize(img, img, img_size);
    
    cv::cvtColor(img, img, cv::COLOR_BGR2BGR565);
    // output the video frame to framebufer row by row
    for (int y = 0; y < img_size.height; y++ )
    {
        fb.seekp(y*fb_info.xres_virtual*fb_info.bits_per_pixel/8);
        fb.write(img.ptr<char>(y, 0), img_size.width*fb_info.bits_per_pixel/8);
    }
    return 0;
}

Mat detect( Mat frame )
{
   std::vector<Rect> faces;
   Mat frame_gray; 
   int eyes_cnt = 0;   

   int scale = 2; 
   // Resize the image to fit screen
   cv::Size2i img_size;
   img_size.height = 600/scale;
   img_size.width = 800/scale;
   double threshold = 80.0;
   int predict_label;
   double predict_confidence;
    
   // Resize the image
   //cv::resize(frame, frame, img_size);

   cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
   equalizeHist( frame_gray, frame_gray );
   //-- Detect faces
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(80, 80) );

   for( size_t i = 0; i < faces.size(); i++ )
    {
      //cout << faces[i] << endl;
      
      //save image for training
      Rect rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
      Mat crop_img = frame_gray(rect);
      //sprintf(img_name, "%d.png", img_cnt++);
      //cv::imwrite(img_name, crop_img);
      
      //predict peopele
      cv::resize(crop_img, crop_img, Size(300,300));
      model->predict(crop_img, predict_label, predict_confidence);
      cv::Point origin;
      origin.x = faces[i].x;
      origin.y = faces[i].y;

      
      if (predict_label == 0){
	   putText(frame,"teamer0",origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, 8, 0);
           //cout << "teamer0 "  << predict_confidence << endl;
      }
      else if(predict_label == 1){
	   putText(frame,"teamer1",origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, 8, 0);
           //cout << "teamer1 "  << predict_confidence << endl;
      }
      else{
           putText(frame,"unknown",origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, 8, 0);
	   //cout << "unknown" << predict_confidence<< endl;
      }
      
      // paint face circle 
      Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
      ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
      
      Mat faceROI = frame_gray( faces[i] );

      cv::Size2i img_size;
      img_size.height = faces[i].height;
      img_size.width = faces[i].width;
      cv::resize(faceROI, faceROI, img_size);

      std::vector<Rect> eyes;

      //-- In each face, detect eyes
      eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(faces[i].width/4,faces[i].height/4));
      eyes_cnt += eyes.size();

      for( size_t j = 0; j < eyes.size(); j++ )
       {
         Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
         int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
         circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
       }
       
    }
    
    /*if(eyes_cnt == faces.size() * 2 && faces.size() != 0){
        cout << (clock() - now)/1000 << endl;
    }*/
    return frame;
}
