kernel_pyramid_blend: kernel_pyramid_blend.cpp
	g++ $< -g -O0 -lGLESv2 -lEGL -lopencv_dnn -lopencv_ml -lopencv_objdetect -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_core -o $@

clean:
	rm -rf kernel_pyramid_blend
