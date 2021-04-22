from ram_utils.pytorch_utils.pytorch_utils import CustomDataSet,preview_img,save_ckp,load_ckp,use_cuda


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torchsummary import summary
from tqdm import tqdm


device = use_cuda
torch.backends.cudnn.benchmark = True



#Writer will output to ./runs/ directory by default.
if not os.path.exists('./cust_dc_img'):
    os.mkdir('./cust_dc_img')

def create_dir(dir_name):
    os.makedirs(dir_name) if not os.path.exists(dir_name) else print("Directory Exist...\n Continue Trainning..")




#for validation 48 img
ano_validation = "/home/pasonatech/workspace/deepNN_py/data_set/anomoly_val_data"
inference_test_data = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/inference_result_aug/"



model_dir = os.path.join("/home/pasonatech/workspace/deepNN_py/carpet_script","model_dir")
create_dir(model_dir)
good_test_dir = "/home/pasonatech/workspace/deepNN_py/data_set/test/good"


# ckp_path = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/model_dir/checkpoint_799.pth"

ckp_path = "/home/pasonatech/workspace/deepNN_py/carpet_script/bw_convert_train/aug_backup4-21/model_dir/checkpoint_799.pth"
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.RandomHorizontalFlip(p=0),
    transforms.RandomVerticalFlip(p=0),
    transforms.RandomRotation(0),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ],inplace=True),
    
    
])


#parameters 
params = {
    'batch_size' : 1,
    'shuffle': False,
    'num_workers': 0
}




class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(                                                   # 3,  1024, 1024
            nn.Conv2d(in_channels = 3, out_channels= 8, kernel_size = 3, padding=1),   #b 8 1024 1024
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size = 2, stride=1,padding=1),
            nn.Conv2d(in_channels = 8, out_channels= 16, kernel_size = 3, padding=1),  # in 16  
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size = 2, stride=2),  # 16 * 512 512
            nn.Conv2d(in_channels = 16, out_channels= 8, kernel_size = 2, padding=1),  # in 16  
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size = 2, stride=1),  # 16 *256
            nn.Conv2d(in_channels = 8, out_channels= 4, kernel_size = 2, padding=1),  # in 16  
            nn.ReLU(True),
            # nn.MaxPool2d(kernel_size = 2, stride=2),  # 16 *256
            #flatten 
            # nn.Flatten(),
            # nn.Linear(in_features = np.product([16,256,256]),out_features=1000)
            
           


            #dense layer
        )
        self.decoder = nn.Sequential(
            #dense layer
            # nn.Linear(in_features=1000,out_features=np.product([16,512,512])),
            # View([-1,16,512,512]),
           
            nn.ConvTranspose2d(4, 8, 2,padding=1),  #４ 8
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 2, padding=1),  # 16 8
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, padding=1),  # 16 8
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=1,padding=1),   # 8 3 
            nn.Tanh()
            #nn.Sigmoid()

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

start_epoch=0
model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
if os.path.exists(ckp_path):
    model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)



# noise = np.random.normal(0, .1, original.shape)
summary(model,(3,1024,1024))

#load one image at a time:
dir_name =good_test_dir

for i in tqdm(sorted(os.listdir(dir_name))):
    # print(i)
    each_img_path = os.path.join(dir_name,i)
    print(each_img_path)
    cust_load_img = CustomDataSet(img_path_or_dir=each_img_path,img_transform=img_transform)
    inferencedataloader = DataLoader(cust_load_img,**params)
    all_inf_loss = []
    for data in inferencedataloader:    
        filename, img= data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)      
        # ===================log========================
        val_loss = criterion(output, img)
        # ===================backward====================
        #write loss function on tensorboard
        print('Inference Loss [{:4f}]'.format(val_loss))
        all_inf_loss.append(val_loss)

        pic = output.cpu().detach()
        preview_img(pic,filename,os.path.join(inference_test_data,"good_test_data"))
    # print(all_inf_loss)




