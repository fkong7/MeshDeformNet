#Copyright (C) 2021 Fanwei Kong, Shawn C. Shadden, University of California, Berkeley

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import os
import glob
class DataLoader:
    #This is a class that loads in the filenames of the image volume and mask volume
    def __init__(self, modality, data_folder, fn='_test', fn_mask='_test_masks', ext='*.nii.gz', ext_out='*.nii.gz'):
        self.modality = modality
        self.data_folder = data_folder
        self.test_fn = fn
        self.test_mask_fn = fn_mask
        self.x_filenames = None
        self.y_filenames = None
        self.ext = ext
        self.ext_out = ext_out
    def set_modality(self, modality):
        self.modality = modality
    def set_datafolder(self, data_folder):
        self.data_folder = data_folder
    def load_datafiles(self):
        x_train_filenames = []
        y_train_filenames = []
        print(os.path.join(self.data_folder,self.modality+self.test_fn))
        for subject_dir in sorted(glob.glob(os.path.join(self.data_folder,self.modality+self.test_fn,self.ext))):
            x_train_filenames.append(os.path.realpath(subject_dir))
        try:
            for subject_dir in sorted(glob.glob(os.path.join(self.data_folder ,self.modality+self.test_mask_fn, self.ext_out))):
                y_train_filenames.append(os.path.realpath(subject_dir))
        except Exception as e: print(e)
        if len(y_train_filenames)==0:
            y_train_filenames = [None]*len(x_train_filenames)
        print("Number of testing volumes %d" % len(x_train_filenames))
        print("Number of mask volumes %d" % len(y_train_filenames))
        self.x_filenames = x_train_filenames
        self.y_filenames = y_train_filenames
        return self.x_filenames, self.y_filenames
    def get_datafiles(self):
        if self.x_filenames is None:
            self.load_datafiles()
        return self.x_filenames, self.y_filenames


