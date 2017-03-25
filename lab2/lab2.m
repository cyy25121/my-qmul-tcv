%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%coursework: face recognition with eigenfaces

%% need to replace with your own path
addpath software;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Loading of the images: You need to replace the directory
Imagestrain = loadImagesInDirectory ( 'dataset/training-set/23x28/');
[Imagestest, Identity] = loadTestImagesInDirectory ( 'dataset/testing-set/23x28/');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of the mean, the eigenvalues, amd the eigenfaces stored in the
%facespace:
ImagestrainSizes = size(Imagestrain);
Means = floor(mean(Imagestrain));
CenteredVectors = (Imagestrain - repmat(Means, ImagestrainSizes(1), 1));

CovarianceMatrix = cov(CenteredVectors);

[U, S, V] = svd(CenteredVectors);
Space = V(: , 1 : ImagestrainSizes(1))';
Eigenvalues = diag(S);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Display of the mean image:
MeanImage = uint8 (zeros(28, 23));
for k = 0:643
   MeanImage( mod (k,28)+1, floor(k/28)+1 ) = Means (1,k+1);

end
figure;
subplot (1, 1, 1);
imshow(MeanImage);
title('Mean Image');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Display of the 20 first eigenfaces : Write your code here

colormap gray;
for k=1:20
    subplot(4,5,k);
    imagesc(reshape(V(:,k), [28, 23]));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Projection of the two sets of images omto the face space:
Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);

Threshold =20;

TrainSizes=size(Locationstrain);
TestSizes = size(Locationstest);
Distances=zeros(TestSizes(1),TrainSizes(1));
%Distances contains for each test image, the distance to every train image.

for i=1:TestSizes(1),
    for j=1: TrainSizes(1),
        Sum=0;
        for k=1: Threshold,
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end,
     Distances(i,j)=Sum;
    end,
end,

Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
for i=1:70,
[Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end,


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Display of first 6 recognition results, image per image:
figure;
x=6;
y=2;
for i=1:6,
      Image = uint8 (zeros(28, 23));
      for k = 0:643
     Image( mod (k,28)+1, floor(k/28)+1 ) = Imagestest (i,k+1);
      end,
   subplot (x,y,2*i-1);
    imshow (Image);
    title('Image tested');

    Imagerec = uint8 (zeros(28, 23));
      for k = 0:643
     Imagerec( mod (k,28)+1, floor(k/28)+1 ) = Imagestrain ((Indices(i,1)),k+1);
      end,
     subplot (x,y,2*i);
imshow (Imagerec);
title(['Image recognised with ', num2str(Threshold), ' eigenfaces:',num2str((Indices(i,1))) ]);
end,



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% recognition rate compared to the number of test images: Write your code here to compute the recognition rate using top 20 eigenfaces.

number_of_test_images=zeros(1,40);
number_of_reg_test_images=zeros(1,40);
recognitionrate=zeros(1,40);
for k=1:70
number_of_test_images(1,Identity(1,k))= number_of_test_images(1,Identity(1,k))+1;%YY I modified here
[Values(k,:), Indices(k,:)] = sort(Distances(k,:));
end

for k=1:70
    id=Identity(1,k);
    if(id==floor((Indices(k,1)-1)/5)+1)
        number_of_reg_test_images(id)=number_of_reg_test_images(id)+1;
    end
end

for k=1:40
    if(number_of_test_images(k) ~= 0)
        recognitionrate(k)=number_of_reg_test_images(k)/number_of_test_images(k);
    else
        recognitionrate(k)=0;
    end
end

recognitionrate_all=sum(number_of_reg_test_images)/sum(number_of_test_images);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% effect of threshold (i.e. number of eigenfaces):
averageRR=zeros(1,20);
for t=1:20,

  Threshold =t;
Distances=zeros(TestSizes(1),TrainSizes(1));

for k=1:TestSizes(1),
    for j=1: TrainSizes(1),
        Sum=0;
        for k=1: Threshold,
   Sum=Sum+((Locationstrain(j,k)-Locationstest(k,k)).^2);
        end,
     Distances(k,j)=Sum;
    end,
end,

Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
number_of_test_images=zeros(1,40);% Number of test images of one given person.%YY I modified here
for k=1:70,
number_of_test_images(1,Identity(1,k))= number_of_test_images(1,Identity(1,k))+1;%YY I modified here
[Values(k,:), Indices(k,:)] = sort(Distances(k,:));
end,

recognised_person=zeros(1,40);
recognitionrate=zeros(1,5);
number_per_number=zeros(1,5);


k=1;
while (k<70),
    id=Identity(1,k);
    distmin=Values(id,1);
        indicemin=Indices(id,1);
    while (k<70)&&(Identity(1,k)==id),
        if (Values(k,1)<distmin),
            distmin=Values(k,1);
        indicemin=Indices(k,1);
        end,
        k=k+1;

    end,
    recognised_person(1,id)=indicemin;
    number_per_number(number_of_test_images(1,id))=number_per_number(number_of_test_images(1,id))+1;
    if (id==floor((indicemin-1)/5)+1) %the good personn was recognised
        recognitionrate(number_of_test_images(1,id))=recognitionrate(number_of_test_images(1,id))+1;

    end,


end,

for  k=1:5,
   recognitionrate(1,k)=recognitionrate(1,k)/number_per_number(1,k);
end,
averageRR(1,t)=mean(recognitionrate(1,:));
end,
figure;
plot(averageRR(1,:));
title('Recognition rate against the number of eigenfaces used');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% effect of K: You need to evaluate the effect of K in KNN and plot the recognition rate against K. Use 20 eigenfaces here.

Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);
yTrain = floor(((1:200)-1)/5+1);
recognitionrate=zeros(1,40);
for k=1:40
    yPredict=knnclassify(Locationstest(:,1:20),Locationstrain(:,1:20),yTrain',k);
    recognitionrate(k)=sum(yPredict' == Identity) / numel(Identity);
end
plot(recognitionrate(1,:));
