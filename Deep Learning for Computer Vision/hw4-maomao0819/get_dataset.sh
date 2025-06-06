
if [ ! -f ./hw4_data.zip ]; then
  # Download dataset
  gdown 'https://drive.google.com/u/0/uc?id=1Tc0f28syYVE185Z6388DWUOVHGSANCmX&export=download' 
fi

# Unzip the downloaded zip file
unzip ./hw4_data.zip