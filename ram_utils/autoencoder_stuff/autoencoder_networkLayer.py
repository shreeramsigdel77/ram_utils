from torch import nn
#new network layers
# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(                                                   # 3,  1024, 1024
#             nn.Conv2d(in_channels = 3, out_channels= 16, kernel_size = 3, padding=1),   #b 8 1024 1024
#             nn.ReLU(True),
#             # nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.MaxPool2d(kernel_size = 4, stride=1),
#             nn.Dropout2d(0.2),

#             nn.Conv2d(in_channels = 16, out_channels= 8, kernel_size = 3, padding=1),  # in 16  
#             nn.ReLU(True),
#             # nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.MaxPool2d(kernel_size = 4, stride=1),  # 16 * 512 512
#             nn.Dropout2d(0.2),

#             nn.Conv2d(in_channels = 8, out_channels= 3, kernel_size = 3, padding=1),  # in 16  
#             nn.ReLU(True),
#             # nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.MaxPool2d(kernel_size = 2, stride=1),  # 16 *256
            
#             # nn.MaxPool2d(kernel_size = 2, stride=2),  # 16 *256
#             #flatten 
#             # nn.Flatten(),
#             # nn.Linear(in_features = np.product([4,62,62]),out_features=8000)
            
           


#             #dense layer
#         )
#         self.decoder = nn.Sequential(
#             #dense layer
#             # nn.Linear(in_features=8000,out_features=np.product([4,62,62])),
#             # View([-1,4,62,62]),
           
#             nn.ConvTranspose2d(3, 8, 3,padding=1,stride=1),  #４ 8
#             nn.ReLU(True),
#             # nn.Upsample(2),
            
#             # nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#             nn.ConvTranspose2d(8, 16, 3, stride=1,padding=1),  # 16 8
#             nn.ReLU(True),
#             # nn.Upsample(4),
#             # nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#             nn.ConvTranspose2d(16, 3, 8, stride=1),  # 16 8
#             nn.ReLU(True),
#             # nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.Tanh()
#             # nn.Sigmoid()

#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         # print(x.shape)
#         # exit()
#         x = self.decoder(x)
#         # print(x.shape)
#         # exit()
#         return x



#################################################################
######################################################################
####################################################################
#linear
import numpy as np
class View(nn.Module):
    def  __init__(self,shape):
        super().__init__()
        self.shape = shape,
    def forward(self,x):
        return x.view(*self.shape)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(    
            nn.Flatten(),
            nn.Linear(in_features =4096 ,out_features=6144),                                             
            nn.ReLU(True),
            
            nn.Dropout(0.2),


            nn.Linear(in_features =6144 ,out_features=8192),                                             
            nn.ReLU(True),
            
            # nn.Dropout(0.2),

            # nn.Linear(in_features =8192 ,out_features=1024),                                             
            # nn.ReLU(True),
            
                   
                    


            #dense layer
        )
        self.decoder = nn.Sequential(
            #dense layer
            # nn.Linear(in_features=8000,out_features=np.product([4,62,62])),
            # View([-1,4,62,62]),
           
            # nn.Linear(in_features =1024 ,out_features=2048),                                             
            # nn.ReLU(True),


            nn.Linear(in_features =8192 ,out_features=6144),                                             
            nn.ReLU(True),

            nn.Linear(in_features =6144 ,out_features=4096),                                             
            nn.ReLU(True),

            # nn.Linear(in_features =10000 ,out_features=16384),                                             
            # nn.ReLU(True),
            
            View([-1,1,64,64])

            # nn.Tanh()
            # nn.Sigmoid()

        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        # exit()
        x = self.decoder(x)
        # print(x.shape)
        # exit()
        return x





#new parameters
# import numpy as np
# class View(nn.Module):
#     def  __init__(self,shape):
#         super().__init__()
#         self.shape = shape,
#     def forward(self,x):
#         return x.view(*self.shape)


# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(    
#             nn.Flatten(),
#             nn.Linear(in_features =4096 ,out_features=3072),                                             
#             nn.ReLU(True),
            
#             nn.Dropout(0.2),


#             nn.Linear(in_features =3072 ,out_features=2048),                                             
#             nn.ReLU(True),
            
#             nn.Dropout(0.2),

#             nn.Linear(in_features =2048 ,out_features=1024),                                             
#             nn.ReLU(True),
            
                   
                    


#             #dense layer
#         )
#         self.decoder = nn.Sequential(
#             #dense layer
#             # nn.Linear(in_features=8000,out_features=np.product([4,62,62])),
#             # View([-1,4,62,62]),
           
#             nn.Linear(in_features =1024 ,out_features=2048),                                             
#             nn.ReLU(True),


#             nn.Linear(in_features =2048 ,out_features=3072),                                             
#             nn.ReLU(True),

#             nn.Linear(in_features =3072 ,out_features=4096),                                             
#             nn.ReLU(True),

#             # nn.Linear(in_features =10000 ,out_features=16384),                                             
#             # nn.ReLU(True),
            
#             View([-1,1,64,64])

#             # nn.Tanh()
#             # nn.Sigmoid()

#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         # print(x.shape)
#         # exit()
#         x = self.decoder(x)
#         # print(x.shape)
#         # exit()
#         return x






#new 

# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(    
#             nn.Flatten(),
#             nn.Linear(in_features =4096 ,out_features=3072),                                             
#             nn.ReLU(True),
            
#             nn.Dropout(0.2),


#             nn.Linear(in_features =3072 ,out_features=2048),                                             
#             nn.ReLU(True),
            
#             nn.Dropout(0.2),

#             nn.Linear(in_features =2048 ,out_features=1024),                                             
#             nn.ReLU(True),
            
                   
                    


#             #dense layer
#         )
#         self.decoder = nn.Sequential(
#             #dense layer
#             # nn.Linear(in_features=8000,out_features=np.product([4,62,62])),
#             # View([-1,4,62,62]),
           
#             nn.Linear(in_features =1024 ,out_features=2048),                                             
#             nn.ReLU(True),


#             nn.Linear(in_features =2048 ,out_features=3072),                                             
#             nn.ReLU(True),

#             nn.Linear(in_features =3072 ,out_features=4096),                                             
#             nn.ReLU(True),

#             # nn.Linear(in_features =10000 ,out_features=16384),                                             
#             # nn.ReLU(True),
            
#             View([-1,1,64,64])

#             # nn.Tanh()
#             # nn.Sigmoid()

#         )
