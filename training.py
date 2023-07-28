if __name__ == "__main__":
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from torchvision.models import segmentation
    from torch.utils.data import Dataset, DataLoader, random_split
    import pandas as pd
    import os
    import tifffile as tiff
    import cv2
    import pandas as pd
    import tifffile as tiff
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    import os
    import torch.nn as nn
    import torch.nn.functional as F


    def calculate_iou(y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred)
        print('intersection----',np.sum(intersection))
        

        union = np.logical_or(y_true, y_pred)
        print('Union----',np.sum(union))
        iou_score = np.sum(intersection) / np.sum(union)
        print('iou score --', iou_score)

        return iou_score
    
    pretrained = True
    if pretrained:
        class CustomDataset(Dataset):
            def __init__(self, csv_file, image_folder, target_folder,transform_image=None,transform_target=None):
                
                self.data = pd.read_csv(csv_file)
                
                
                #the scene input images and the target segmentation maps both have the same name but in different folder
                
                
                
                self.image_paths = self.data['file']
                self.image_paths = self.image_paths[:50] #for testing and debugging
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
                image= image[:,:,:3]/10000
                #print('1----image shape ----->', image.shape)
                #print(type(image))
                #image = torch.from_numpy(image.astype('float32'))
                #image = torch.from_numpy(image.astype(np.float32))


                




                image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_AREA)
                #print('3 -- after resize---image shape ----->', image.shape) 
                #print('max------------',np.max(image))
                #
                #image = np.transpose(image, (2,0,1))
                #print('2----after transpose  image shape ----->', image.shape)

                target_path = os.path.join(self.target_folder, image_name)
                target = tiff.imread(target_path)
                target = np.transpose(target, (1,2,0))





                target = cv2.resize(target, dsize=(512, 512), interpolation=cv2.INTER_AREA)

                #target = target.astype('uint8')
                
                ## turning the ground truth segmenation mask into a (L,W,CLASSES) one hot-hot encoded matrix
                #target = Image.fromarray(target)
                #target = target.resize((512,512))
                
                
                

                if self.transform:
                    
                    image = self.transform(image)
                    target = self.transform_target(target)
                    
                    
                    
                #print('after transofrm image shape --',image.shape)   
                #print('max------------',torch.max(image).item())
                #print('min------------',torch.min(image).item())
                #print('max of target------------',torch.max(target).item())
                #print('min of target------------',torch.min(target).item())
                return image, target

            def __len__(self):
                return len(self.image_paths)
            #-----------------------------------------------------------------------------------#

        # Define the paths to your training data


        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Initialize the model and set it to training mode

        model = segmentation.deeplabv3_resnet50(pretrained=False)
        model.classifier[-1] = torch.nn.Conv2d(256, 43, kernel_size=1)
        checkpoint_path = '/home/ahmedemam576/greybox/corine_images/pretrained_ep48_BS_4.pth'  # Replace with the actual path to your saved checkpoint file

        #model.load_state_dict(torch.load(checkpoint_path))
        model.train()
        model.to('cuda')

        # Set up the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Define the transformation for input images and labels
        transform = transforms.Compose([
            #transforms.ToPILImage(),
            
            #transforms.Resize((512,512)),
            transforms.ToTensor()
            
            
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
        batch_size = 4
        # Create the data loader
        #dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True, drop_last=True)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio

        # Calculate the number of samples for each split
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        # Perform the split
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create data loaders for training and testing
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last= True)





        # Training loop
        num_epochs = 100
        for epoch in tqdm(range(num_epochs), desc="Main training"):

            for images, targets in tqdm(dataloader, desc="Current epoch", leave = False):
                model.train()
                optimizer.zero_grad()
                images, targets = images.cuda().float(), targets.cuda()
                # Forward pass
                outputs = model(images)        
                loss = criterion(outputs['out'], targets)

                # Backward pass and optimization
                loss.backward()


                optimizer.step()
              
                
            if epoch%2:
                iou_score =0
                predicted_masks = (outputs['out'] > 0.5)
                
                targets_ = targets.detach().cpu().numpy()
                outputs_ =outputs['out'].detach().cpu().numpy()
                intersection = np.logical_and(targets_, outputs_)
                print('intersection----',np.sum(intersection))
                

                union = np.logical_or(targets_, outputs_)
                print('Union----',np.sum(union))
                iou_score +=  np.sum(intersection) / np.sum(union)

                print('iou score --', iou_score)




                #iou = calculate_iou(predicted_masks, targets)
                #print(f"Epoch {epoch + 1}/{num_epochs} - IOU : {iou:.4f} ")
                torch.save(model.state_dict(), f'newmodel{epoch}.pth')

            if epoch%50 == 0:
                print('--------- evaluation step -------------')
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_iou = 0.0
                    model.to('cuda')
                    for imgs, targets in tqdm(test_dataloader):
                        imgs, targets = imgs.to('cuda').float(), targets.to('cuda')
                        outputs = model(imgs)['out']
                        val_loss += criterion(outputs, targets).item()
                        predicted_masks = (outputs > 0.5).cpu().numpy().astype(np.uint8)
                        val_iou += calculate_iou(predicted_masks, targets.cpu().numpy())
                    val_loss /= len(imgs) / batch_size
                    val_iou /= len(imgs) / batch_size
                print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f} - Validation IoU: {val_iou:.4f}")
                


    else:

        
        # Custom dataset class
        class CustomDataset(Dataset):
            def __init__(self, csv_file, image_folder, target_folder,transform_image=None,transform_target=None):
                
                self.data = pd.read_csv(csv_file)
                
                
                #the scene input images and the target segmentation maps both have the same name but in different folder
                
                
                
                self.image_paths = self.data['file']
                #self.image_paths = self.image_paths[:10] for testing and debugging
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





                target = cv2.resize(target, dsize=(512, 512), interpolation=cv2.INTER_AREA)

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
        batch_size = 5
        # Create the data loader
        dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True, drop_last=True)

        # Training loop
        num_epochs = 4
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
            if epoch%2 == 0:
                torch.save(model.state_dict(), f'fine_model_ep{epoch}_BS_{batch_size}.pth')


        # Save the fine-tuned model
        


