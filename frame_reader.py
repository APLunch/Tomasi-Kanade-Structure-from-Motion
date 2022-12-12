import cv2
import os

# a class that takes a directory, read frames of all the .mp4 files, store them in separated folders by their names
class FrameReader:
    def __init__(self, dir):
        self.directory = dir

    def process_one_video(self, filename):
        # Read the video from specified path
        video_path = os.path.join(self.directory, filename)
        cam = cv2.VideoCapture(video_path)
        if not os.path.exists('data'):
                os.makedirs('data')
        
        curdir = os.getcwd()
        os.chdir('data')
        try:
            name = os.path.splitext(filename)[0]
            # creating a folder named data
            dataset_name = 'frames'+'_'+name
            if not os.path.exists(dataset_name):
                os.makedirs(dataset_name)
        
        # if not created then raise error
        except OSError:
            print ('Error: Creating directory of dataset')
        
        # frame
        currentframe = 0
        while(True):
            # reading from frame
            ret, frame = cam.read()
            if ret:
                # if video is still left continue creating images
                if (currentframe%5==0):
                    name = './'+ dataset_name + '/' + dataset_name +str(currentframe) + '.jpg'
                    print ('Creating...' + name)
        
                # writing the extracted images
                cv2.imwrite(name, frame)
        
                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break
        
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

        os.chdir(curdir)

    def read_all_frames(self):
        dirlist = os.listdir(self.directory)
        videolist = [s for s in dirlist if '.mp4' in s]
        for filename in videolist:
            self.process_one_video(filename)


