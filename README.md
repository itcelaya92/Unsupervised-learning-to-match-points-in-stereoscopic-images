# Unsupervised-learning-to-match-points-in-stereoscopic-images
Point Matching using epipolar constraint algorithms and unsupervised machine learning to calculate strain in metal sheets.

This is an improvement of the preceding work https://github.com/itcelaya92/3D-Reconstruction-of-metal-sheet 
Here a method was added to find a first correspondence that serves to trigger the neighborhood search algorithm. 
The new method is the following:
1. Stamping of undeformed sheet metal with a uniform pattern of circles.
2. Calibration of the stereoscopic or multi-camera vision system.
3. Acquisition of images in a controlled lighting environment.

![image](https://user-images.githubusercontent.com/87040483/158498425-79f7f92c-4074-4d4d-a4d2-7d8a5d73cfac.png)
![image](https://user-images.githubusercontent.com/87040483/158498440-7350798c-f568-472c-beef-27439bf4d82a.png)

4. Digital processing of the images to label the circles of the uniform pattern drawn on the surface of the sheet.
![image](https://user-images.githubusercontent.com/87040483/158498367-22835da6-852e-4b92-9388-a06ef0298287.png)

5. Matching points:

  a. DBSCAN to group the labels that are in the same epiline.
  
  ![image](https://user-images.githubusercontent.com/87040483/158498267-b8afcf16-a852-49b3-b330-7db5dacd00db.png)
  ![image](https://user-images.githubusercontent.com/87040483/158498624-56cd5828-ade2-4f4f-b25b-3fc1e83b4de9.png)
  ![image](https://user-images.githubusercontent.com/87040483/158498847-5cf0b0c9-cb3d-4d24-8a11-6979c499a34a.png)


  b. Implementation of the matching algorithm by the disparity in each group of labels that belong to the same epiline in both images.
  c. Find and match neighbor labels between the left and right images.
  
6. Triangulation of points to obtain their position in 3D space and reconstruction of the metal sheet.

![image](https://user-images.githubusercontent.com/87040483/158498893-687e9062-caf3-46f2-baab-377e30b793cc.png)

7. Calculation of the deformation at each point of the piece.

![image](https://user-images.githubusercontent.com/87040483/158498940-8677cb33-b945-4d57-8196-5475ef783225.png)
![image](https://user-images.githubusercontent.com/87040483/158498950-c48fca96-4b5a-491e-8b18-4cb5a4928317.png)
