if __name__ == "__main__":

    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from torchvision.models import segmentation
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd
    import os
    import tifffile as tiff
    import cv2
    from tqdm import tqdm
    import numpy as np

    # Custom dataset class
    class CustomDataset(Dataset):
        def __init__(self, csv_file, image_folder, target_folder,transform_image=None,transform_target=None):
            
            self.data = pd.read_csv(csv_file)
            
            
            #the scene input images and the target segmentation maps both have the same name but in different folder
            
            
            
            self.image_paths = self.data['file']
            self.image_paths = self.image_paths[:10]
            self.image_folder = image_folder
            self.target_folder = target_folder
            
            
            self.transform = transform
            self.transform_target = transform_target

        def __getitem__(self, index):
    

            
            # Construct the complete image path by joining the folder path and image name
            image_name = self.image_paths[index] 
            image_path = os.path.join(self.image_folder, image_name)

            
            # Open the image using PIL
            image = tiff.imread(image_path)
            #choos ethe number of channels you need
            image= image[:,:,:3]

            image = image.astype('uint8')
            
            #
            
            target_path = os.path.join(self.target_folder, image_name)
            target = tiff.imread(target_path)
            target = np.transpose(target, (1,2,0))





            target = cv2.resize(target, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

            #target = target.astype('uint8')
            
            ## turning the ground truth segmenation mask into a (L,W,CLASSES) one hot-hot encoded matrix
            #target = Image.fromarray(target)
            #target = target.resize((512,512))
            
            
            

            if self.transform:
                
                image = self.transform(image)
                target = self.transform_target(target)
                
                
                

            return image, target

        def __len__(self):
            return len(self.image_paths)
        #-----------------------------------------------------------------------------------#

    # Define the paths to your training data


    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize the model and set it to training mode

    model = segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(256, 43, kernel_size=1)
    model.train()
    model.to('cuda')

    # Set up the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Define the transformation for input images and labels
    transform = transforms.Compose([
        transforms.ToPILImage(),
        
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    ])

    #transforms.ToPILImage(),
    #transform_target =transforms.Compose([transforms.Resize((512, 512)),
    #                                         transforms.ToTensor()])

    transform_target = transforms.ToTensor()

    #---------------------------------------------------------------------------------------------------#

    csv_file =  'infos.csv'
    image_folder = '/home/ahmedemam576/working_folder/data/anthroprotect/tiles/s2'
    target_folder='/home/ahmedemam576/greybox/corine_images/onehotcoded_masks/'
        
    # Create the custom dataset
    dataset = CustomDataset(csv_file, image_folder, target_folder,transform_image=transform, transform_target =transform_target)

    # Create the data loader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        for images, targets in tqdm(dataloader):
            
            optimizer.zero_grad()
            images, targets = images.cuda(), targets.cuda()
            # Forward pass
            outputs = model(images)        
            loss = criterion(outputs['out'], targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'fine_tuned_model.pth')


