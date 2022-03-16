import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
import cv2

with np.load('Valores_Calibracion.npz') as X:
    cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, R, T, E, F, R1, R2, P1, P2, Q = [X[i] for i in ('cameraMatrix1', 'cameraMatrix2', 'distCoeffs1', 'distCoeffs2', 'R', 'T', 'E', 'F', 'R1', 'R2', 'P1', 'P2', 'Q')]
with np.load('Etiquetas.npz') as X:
    labels_R, labels, centroids_R, centroids, num_R, num = [X[i] for i in ('labels_R', 'labels', 'centroids_R', 'centroids', 'num_R', 'num')]


left_maps = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (2752, 2200), cv2.CV_16SC2)
right_maps = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (2752, 2200), cv2.CV_16SC2)
pts_Left = np.array([(centroids[0][0], centroids[0][1])])
pts_Right = np.array([(centroids_R[0][0], centroids_R[0][1])])

for a in range(1, centroids_R.__len__()):
    pto = centroids_R[a][0]
    pto1 = centroids_R[a][1]
    pts_Conca = np.array([(pto, pto1)])
    pts_Right = np.concatenate((pts_Right, pts_Conca))

for a in range(1, centroids.__len__()):
    pto = centroids[a][0]
    pto1 = centroids[a][1]
    pts_Conca = np.array([(pto, pto1)])
    pts_Left = np.concatenate((pts_Left, pts_Conca))


pts_Left = np.float32(pts_Left[:, np.newaxis, :])
pts_Right = np.float32(pts_Right[:, np.newaxis, :])

pts_und_Left = cv2.undistortPoints(pts_Left, cameraMatrix1, distCoeffs1, R=R1, P=P1 )
pts_und_Right = cv2.undistortPoints(pts_Right, cameraMatrix2, distCoeffs2, R=R2, P=P2)

# Map component labels to hue val
label_hue = np.uint8(179 * labels_R / np.max(labels_R))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img_R = cv2.merge([label_hue, blank_ch, blank_ch])
# cvt to BGR for display
labeled_img_R = cv2.cvtColor(labeled_img_R, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img_R[label_hue == 0] = 0

# Map component labels to hue val
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img[label_hue == 0] = 0

cv2.imwrite('CircleR_R.png', labeled_img_R)
cv2.imwrite('CircleL_L.png', labeled_img)
lFrame = cv2.imread('CircleL_L.png')
rFrame = cv2.imread('CircleR_L.png')

left_img_remap = cv2.remap(lFrame, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
right_img_remap = cv2.remap(rFrame, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)


with np.load('Fila.npz') as X:
    R_Fila, L_Fila = [X[i] for i in ('R_Fila2', 'L_Fila2')]
print(R_Fila)

#Para imagen derecha
R_Fila = StandardScaler().fit_transform(R_Fila)
b, c, d = np.vsplit(R_Fila, 3)
imagenDere = []

#epsilon = .0023423433303833005644091

#para b Right
epsilon = .0023423433303833005644091
minimumSamples = 5
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(b)
labelsA = db.labels_
labelsA
#print(labelsA[5])
#print(labelsA.__len__(), labelsA.__len__()-db.core_sample_indices_.__len__()) #no hay separacion entre lineas, puede que si

# Primer, crear un vector de valores booleanos (valores binarios (verdadero/falso)) usando las etiquetas de la variable db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
#print(core_samples_mask)

# Número de clusters en etiquetas, ignorando el ruido en caso de estar presente.
n_clusters_1R = len(set(labelsA)) - (1 if -1 in labelsA else 0)
n_clusters_1R
print(n_clusters_1R)

# Eliminar la repetición en las etiquetas transformándolas en un conjunto.
unique_labels = set(labelsA)
unique_labels
#print(unique_labels)

# Crear colores para los clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors

count = 0
right_img_remap2 = right_img_remap.copy()

#Dibujar los puntos con colores
for k, col in zip(unique_labels, colors):
    if k == -1:
        #Black used for noise.
        col = [0, 0, 0, 1]
        
    class_member_mask = (labelsA == k)
    
    # Dibujoar los datos que estan agrupados (clusterizados)
    xy = b[class_member_mask & core_samples_mask]
    #print(xy.__len__(),xy, class_member_mask & core_samples_mask)
    plt.plot(xy, 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
   
    Indices = [class_member_mask & core_samples_mask][0]
    Indices2 = [i for i, e in enumerate(Indices)  if e == True]
    
    R = 0
    G = 0
    B = 255
    
    coordX = 0
    count_ejey = 0
    for indic in Indices2:
        count_ejey = count_ejey+1
        cv2.circle(right_img_remap, (int(pts_und_Right[indic][0][0]), int(pts_und_Right[indic][0][1])), 1, (R,G,B), 2)
        coordX = int(pts_und_Right[indic][0][1]) + coordX
        coordX1 = int(coordX /  count_ejey)
        
    ImagenR_Fila =  np.array(coordX1)   
    ImagenR_Fila =  np.append(ImagenR_Fila, Indices2)
    #print(ImagenR_Fila)
    
    # Dibujar los valores atípicos
    Indices3 = [class_member_mask & ~core_samples_mask][0]
    Indices4 = [i for i, e in enumerate(Indices3)  if e == True]
    
    for indic in Indices4:
        cv2.circle(right_img_remap, (int(pts_und_Right[indic][0][0]), int(pts_und_Right[indic][0][1])), 1, tuple(col), 2)
    
    #xy = b[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    #cv2.imwrite('fila_each{}.jpg'.format(count), right_img_remap)
    count = count + 1
    
    right_img_remap = right_img_remap2.copy()
    
    imagenDere.append(ImagenR_Fila)
    ImagenR_Fila = []
    
    
#Para c Right
epsilon = .00234229
minimumSamples = 5
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(c)
labelsA = db.labels_
labelsA

# Primer, crear un vector de valores booleanos (valores binarios (verdadero/falso)) usando las etiquetas de la variable db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
#print(core_samples_mask)

# Número de clusters en etiquetas, ignorando el ruido en caso de estar presente.
n_clusters_2R = len(set(labelsA)) - (1 if -1 in labelsA else 0)
n_clusters_2R
print(n_clusters_2R)

# Eliminar la repetición en las etiquetas transformándolas en un conjunto.
unique_labels = set(labelsA)
unique_labels
#print(unique_labels)

# Crear colores para los clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors

right_img_remap2 = right_img_remap.copy()
#print(right_img_remap.__len__())

#Dibujar los puntos con colores
for k, col in zip(unique_labels, colors):
    if k == -1:
        #Black used for noise.
        col = [0, 0, 0, 1]
        
    class_member_mask = (labelsA == k)
    
    # Dibujoar los datos que estan agrupados (clusterizados)
    xy = c[class_member_mask & core_samples_mask]
    #print(xy.__len__(),xy, class_member_mask & core_samples_mask)
    plt.plot(xy, 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    
    Indices = [class_member_mask & core_samples_mask][0]
    Indices2 = [i for i, e in enumerate(Indices)  if e == True]
    Indices3 = []
    for indice in Indices2:
        indice = indice + 4469
        Indices3.append(indice)
    
    R = 0
    G = 0
    B = 255
    
    coordX = 0
    count_ejey = 0
    for indic in Indices3:
        count_ejey = count_ejey+1
        cv2.circle(right_img_remap, (int(pts_und_Right[indic][0][0]), int(pts_und_Right[indic][0][1])), 1, (R,G,B), 2)
        coordX = int(pts_und_Right[indic][0][1]) + coordX
        coordX1 = int(coordX /  count_ejey)
        
    ImagenR_Fila =  np.array(coordX1)       
    ImagenR_Fila =  np.append(ImagenR_Fila, Indices3)
    
    # Dibujar los valores atípicos
    Indices5 = [class_member_mask & ~core_samples_mask][0]
    Indices4 = [i for i, e in enumerate(Indices5)  if e == True]
    
    for indic in Indices4:
        cv2.circle(right_img_remap, (int(pts_und_Right[indic][0][0]), int(pts_und_Right[indic][0][1])), 1, tuple(col), 2)
    
    #xy = b[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    #cv2.imwrite('fila_each{}.jpg'.format(count), right_img_remap)
    count = count + 1
    
    right_img_remap = right_img_remap2.copy()
    imagenDere.append(ImagenR_Fila)
    ImagenR_Fila = []


#Para d Right
epsilon = .0023423433303833005644091
minimumSamples = 5
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(d)
labelsA = db.labels_
labelsA

# Primer, crear un vector de valores booleanos (valores binarios (verdadero/falso)) usando las etiquetas de la variable db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
#print(core_samples_mask)

# Número de clusters en etiquetas, ignorando el ruido en caso de estar presente.
n_clusters_3R = len(set(labelsA)) - (1 if -1 in labelsA else 0)
n_clusters_3R
print(n_clusters_3R)

# Eliminar la repetición en las etiquetas transformándolas en un conjunto.
unique_labels = set(labelsA)
unique_labels
#print(unique_labels)

# Crear colores para los clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors

right_img_remap2 = right_img_remap.copy()
#print(right_img_remap.__len__())

#Dibujar los puntos con colores
for k, col in zip(unique_labels, colors):
    if k == -1:
        #Black used for noise.
        col = [0, 0, 0, 1]
        
    class_member_mask = (labelsA == k)
    
    # Dibujoar los datos que estan agrupados (clusterizados)
    xy = d[class_member_mask & core_samples_mask]
    #print(xy.__len__(),xy, class_member_mask & core_samples_mask)
    plt.plot(xy, 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    
    Indices = [class_member_mask & core_samples_mask][0]
    Indices2 = [i for i, e in enumerate(Indices)  if e == True]
    Indices3 = []
    for indice in Indices2:
        indice = indice + 8938
        Indices3.append(indice)
    
    R = 0
    G = 0
    B = 255
    
    coordX = 0
    count_ejey = 0
    for indic in Indices3:
        count_ejey = count_ejey+1
        cv2.circle(right_img_remap, (int(pts_und_Right[indic][0][0]), int(pts_und_Right[indic][0][1])), 1, (R,G,B), 2)
        coordX = int(pts_und_Right[indic][0][1]) + coordX
        coordX1 = int(coordX /  count_ejey)
        
    ImagenR_Fila =  np.array(coordX1)   
    ImagenR_Fila =  np.append(ImagenR_Fila, Indices3)
    
    # Dibujar los valores atípicos
    Indices5 = [class_member_mask & ~core_samples_mask][0]
    Indices4 = [i for i, e in enumerate(Indices5)  if e == True]
    
    for indic in Indices4:
        cv2.circle(right_img_remap, (int(pts_und_Right[indic][0][0]), int(pts_und_Right[indic][0][1])), 1, tuple(col), 2)
    
    #xy = b[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    #cv2.imwrite('fila_each{}.jpg'.format(count), right_img_remap)
    count = count + 1
    
    right_img_remap = right_img_remap2.copy()
    imagenDere.append(ImagenR_Fila)
    ImagenR_Fila = []
print(imagenDere.__len__(), imagenDere[29][0])

plt.title('Estimated number of clusters: %d' % (n_clusters_1R + n_clusters_2R + n_clusters_3R))
plt.show()

#Para imagen izquierda
L_Fila = L_Fila[:-1]
L_Fila = L_Fila[:-1]

L_Fila = StandardScaler().fit_transform(L_Fila)
b, c, d, w, r = np.vsplit(L_Fila, 5)
imagenIzq = []
#epsilon = .0023423433303833005644091

#para b Left
epsilon = .00229
minimumSamples = 5
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(b)
labelsA = db.labels_
labelsA
#print(L_Fila.__len__()/5)
#print(labelsA.__len__(), labelsA.__len__()-db.core_sample_indices_.__len__()) #no hay separacion entre lineas, puede que si

# Primer, crear un vector de valores booleanos (valores binarios (verdadero/falso)) usando las etiquetas de la variable db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
#print(core_samples_mask)

# Número de clusters en etiquetas, ignorando el ruido en caso de estar presente.
n_clusters_1L = len(set(labelsA)) - (1 if -1 in labelsA else 0)
n_clusters_1L
print(n_clusters_1L)

# Eliminar la repetición en las etiquetas transformándolas en un conjunto.
unique_labels = set(labelsA)
unique_labels
#print(unique_labels)

# Crear colores para los clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors

count = 0
left_img_remap2 = left_img_remap.copy()
#print(left_img_remap.__len__())

#Dibujar los puntos con colores
for k, col in zip(unique_labels, colors):
    if k == -1:
        #Black used for noise.
        col = [0, 0, 0, 1]
        
    class_member_mask = (labelsA == k)
    
    # Dibujoar los datos que estan agrupados (clusterizados)
    xy = b[class_member_mask & core_samples_mask]
    #print(xy.__len__(),xy, class_member_mask & core_samples_mask)
    plt.plot(xy, 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    
    Indices = [class_member_mask & core_samples_mask][0]
    Indices2 = [i for i, e in enumerate(Indices)  if e == True]
    
    R = 0
    G = 0
    B = 255
    
    coordX = 0
    count_ejey = 0
    for indic in Indices2:
        count_ejey = count_ejey+1
        cv2.circle(left_img_remap, (int(pts_und_Left[indic][0][0]), int(pts_und_Left[indic][0][1])), 1, (R,G,B), 2)
        coordX = int(pts_und_Left[indic][0][1]) + coordX
        coordX1 = int(coordX /  count_ejey)
        
    ImagenL_Fila =  np.array(coordX1)   
    ImagenL_Fila =  np.append(ImagenL_Fila, Indices2)
    
    # Dibujar los valores atípicos
    Indices3 = [class_member_mask & ~core_samples_mask][0]
    Indices4 = [i for i, e in enumerate(Indices3)  if e == True]
    
    for indic in Indices4:
        cv2.circle(left_img_remap, (int(pts_und_Left[indic][0][0]), int(pts_und_Left[indic][0][1])), 1, tuple(col), 2)
    
    #xy = b[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    #cv2.imwrite('fila_each{}.jpg'.format(count), left_img_remap)
    count = count + 1
    
    left_img_remap = left_img_remap2.copy()
    imagenIzq.append(ImagenL_Fila)
    ImagenL_Fila = []

  
#Para c Left
epsilon = .0022899
minimumSamples = 5
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(c)
labelsA = db.labels_
labelsA

# Primer, crear un vector de valores booleanos (valores binarios (verdadero/falso)) usando las etiquetas de la variable db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
#print(core_samples_mask)

# Número de clusters en etiquetas, ignorando el ruido en caso de estar presente.
n_clusters_2L = len(set(labelsA)) - (1 if -1 in labelsA else 0)
n_clusters_2L
print(n_clusters_2L)

# Eliminar la repetición en las etiquetas transformándolas en un conjunto.
unique_labels = set(labelsA)
unique_labels
#print(unique_labels)

# Crear colores para los clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors

left_img_remap2 = left_img_remap.copy()
#print(left_img_remap.__len__())

#Dibujar los puntos con colores
for k, col in zip(unique_labels, colors):
    if k == -1:
        #Black used for noise.
        col = [0, 0, 0, 1]
        
    class_member_mask = (labelsA == k)
    
    # Dibujoar los datos que estan agrupados (clusterizados)
    xy = c[class_member_mask & core_samples_mask]
    #print(xy.__len__(),xy, class_member_mask & core_samples_mask)
    plt.plot(xy, 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    
    Indices = [class_member_mask & core_samples_mask][0]
    Indices2 = [i for i, e in enumerate(Indices)  if e == True]
    Indices3 = []
    for indice in Indices2:
        indice = indice + 2681   #(L_Fila.__len__()/5)
        Indices3.append(indice)
    
    R = 0
    G = 0
    B = 255
    
    coordX = 0
    count_ejey = 0
    for indic in Indices3:
        count_ejey = count_ejey+1
        cv2.circle(left_img_remap, (int(pts_und_Left[indic][0][0]), int(pts_und_Left[indic][0][1])), 1, (R,G,B), 2)
        coordX = int(pts_und_Left[indic][0][1]) + coordX
        coordX1 = int(coordX /  count_ejey)
        
    ImagenL_Fila =  np.array(coordX1)     
    ImagenL_Fila =  np.append(ImagenL_Fila, Indices3)
    
    # Dibujar los valores atípicos
    Indices5 = [class_member_mask & ~core_samples_mask][0]
    Indices4 = [i for i, e in enumerate(Indices5)  if e == True]
    
    for indic in Indices4:
        cv2.circle(left_img_remap, (int(pts_und_Left[indic][0][0]), int(pts_und_Left[indic][0][1])), 1, tuple(col), 2)
    
    #xy = b[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    #cv2.imwrite('fila_each{}.jpg'.format(count), left_img_remap)
    count = count + 1
    
    left_img_remap = left_img_remap2.copy()
    imagenIzq.append(ImagenL_Fila)
    ImagenL_Fila = []

#Para d Left
epsilon = .002289914
minimumSamples = 5
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(d)
labelsA = db.labels_
labelsA

# Primer, crear un vector de valores booleanos (valores binarios (verdadero/falso)) usando las etiquetas de la variable db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
#print(core_samples_mask)

# Número de clusters en etiquetas, ignorando el ruido en caso de estar presente.
n_clusters_3L = len(set(labelsA)) - (1 if -1 in labelsA else 0)
n_clusters_3L
print(n_clusters_3L)

# Eliminar la repetición en las etiquetas transformándolas en un conjunto.
unique_labels = set(labelsA)
unique_labels
#print(unique_labels)

# Crear colores para los clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors

left_img_remap2 = left_img_remap.copy()
#print(right_img_remap.__len__())

#Dibujar los puntos con colores
for k, col in zip(unique_labels, colors):
    if k == -1:
        #Black used for noise.
        col = [0, 0, 0, 1]
        
    class_member_mask = (labelsA == k)
    
    # Dibujoar los datos que estan agrupados (clusterizados)
    xy = d[class_member_mask & core_samples_mask]
    #print(xy.__len__(),xy, class_member_mask & core_samples_mask)
    plt.plot(xy, 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    
    Indices = [class_member_mask & core_samples_mask][0]
    Indices2 = [i for i, e in enumerate(Indices)  if e == True]
    Indices3 = []
    for indice in Indices2:
        indice = indice + 5362 #((R_Fila.__len__()/5)*2)
        Indices3.append(indice)
    
    R = 0
    G = 0
    B = 255
    
    coordX = 0
    count_ejey = 0
    for indic in Indices3:
        count_ejey = count_ejey+1
        cv2.circle(left_img_remap, (int(pts_und_Left[indic][0][0]), int(pts_und_Left[indic][0][1])), 1, (R,G,B), 2)
        coordX = int(pts_und_Left[indic][0][1]) + coordX
        coordX1 = int(coordX /  count_ejey)
        
    ImagenL_Fila =  np.array(coordX1)       
    ImagenL_Fila =  np.append(ImagenL_Fila, Indices3)
    
    # Dibujar los valores atípicos
    Indices5 = [class_member_mask & ~core_samples_mask][0]
    Indices4 = [i for i, e in enumerate(Indices5)  if e == True]
    
    for indic in Indices4:
        cv2.circle(left_img_remap, (int(pts_und_Left[indic][0][0]), int(pts_und_Left[indic][0][1])), 1, tuple(col), 2)
    
    #xy = b[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    #cv2.imwrite('fila_each{}.jpg'.format(count), left_img_remap)
    count = count + 1
    
    left_img_remap = left_img_remap2.copy()
    imagenIzq.append(ImagenL_Fila)
    ImagenL_Fila = []

#Para w Left
epsilon = .002289914
minimumSamples = 5
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(w)
labelsA = db.labels_
labelsA

# Primer, crear un vector de valores booleanos (valores binarios (verdadero/falso)) usando las etiquetas de la variable db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
#print(core_samples_mask)

# Número de clusters en etiquetas, ignorando el ruido en caso de estar presente.
n_clusters_4L = len(set(labelsA)) - (1 if -1 in labelsA else 0)
n_clusters_4L
print(n_clusters_4L)

# Eliminar la repetición en las etiquetas transformándolas en un conjunto.
unique_labels = set(labelsA)
unique_labels
#print(unique_labels)

# Crear colores para los clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors

left_img_remap2 = left_img_remap.copy()
#print(right_img_remap.__len__())

#Dibujar los puntos con colores
for k, col in zip(unique_labels, colors):
    if k == -1:
        #Black used for noise.
        col = [0, 0, 0, 1]
        
    class_member_mask = (labelsA == k)
    
    # Dibujoar los datos que estan agrupados (clusterizados)
    xy = w[class_member_mask & core_samples_mask]
    #print(xy.__len__(),xy, class_member_mask & core_samples_mask)
    plt.plot(xy, 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    
    Indices = [class_member_mask & core_samples_mask][0]
    Indices2 = [i for i, e in enumerate(Indices)  if e == True]
    Indices3 = []
    for indice in Indices2:
        indice = indice + 8043 #((R_Fila.__len__()/5)*3)
        Indices3.append(indice)
    
    R = 0
    G = 0
    B = 255
    
    coordX = 0
    count_ejey = 0
    for indic in Indices3:
        count_ejey = count_ejey+1
        cv2.circle(left_img_remap, (int(pts_und_Left[indic][0][0]), int(pts_und_Left[indic][0][1])), 1, (R,G,B), 2)
        coordX = int(pts_und_Left[indic][0][1]) + coordX
        coordX1 = int(coordX /  count_ejey)
        
    ImagenL_Fila =  np.array(coordX1)        
    ImagenL_Fila =  np.append(ImagenL_Fila, Indices3)
    
    # Dibujar los valores atípicos
    Indices5 = [class_member_mask & ~core_samples_mask][0]
    Indices4 = [i for i, e in enumerate(Indices5)  if e == True]
    
    for indic in Indices4:
        cv2.circle(left_img_remap, (int(pts_und_Left[indic][0][0]), int(pts_und_Left[indic][0][1])), 1, tuple(col), 2)
    
    #xy = b[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    #cv2.imwrite('fila_each{}.jpg'.format(count), left_img_remap)
    count = count + 1
    
    left_img_remap = left_img_remap2.copy()
    imagenIzq.append(ImagenL_Fila)
    ImagenL_Fila = []

#Para r Left
epsilon = .002289914
minimumSamples = 5
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(r)
labelsA = db.labels_
labelsA

# Primer, crear un vector de valores booleanos (valores binarios (verdadero/falso)) usando las etiquetas de la variable db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
#print(core_samples_mask)

# Número de clusters en etiquetas, ignorando el ruido en caso de estar presente.
n_clusters_5L = len(set(labelsA)) - (1 if -1 in labelsA else 0)
n_clusters_5L
print(n_clusters_5L)

# Eliminar la repetición en las etiquetas transformándolas en un conjunto.
unique_labels = set(labelsA)
unique_labels
#print(unique_labels)

# Crear colores para los clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors

left_img_remap2 = left_img_remap.copy()
#print(right_img_remap.__len__())

#Dibujar los puntos con colores
for k, col in zip(unique_labels, colors):
    if k == -1:
        #Black used for noise.
        col = [0, 0, 0, 1]
        
    class_member_mask = (labelsA == k)
    
    # Dibujoar los datos que estan agrupados (clusterizados)
    xy = r[class_member_mask & core_samples_mask]
    #print(xy.__len__(),xy, class_member_mask & core_samples_mask)
    plt.plot(xy, 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    
    Indices = [class_member_mask & core_samples_mask][0]
    Indices2 = [i for i, e in enumerate(Indices)  if e == True]
    Indices3 = []
    for indice in Indices2:
        indice = indice + 10724 #((R_Fila.__len__()/5)*4)
        Indices3.append(indice)
    
    R = 0
    G = 0
    B = 255
    
    coordX = 0
    count_ejey = 0
    for indic in Indices3:
        count_ejey = count_ejey+1
        cv2.circle(left_img_remap, (int(pts_und_Left[indic][0][0]), int(pts_und_Left[indic][0][1])), 1, (R,G,B), 2)
        coordX = int(pts_und_Left[indic][0][1]) + coordX
        coordX1 = int(coordX /  count_ejey)
        
    ImagenL_Fila =  np.array(coordX1)     
    ImagenL_Fila =  np.append(ImagenL_Fila, Indices3)
    
    # Dibujar los valores atípicos
    Indices5 = [class_member_mask & ~core_samples_mask][0]
    Indices4 = [i for i, e in enumerate(Indices5)  if e == True]
    
    for indic in Indices4:
        cv2.circle(left_img_remap, (int(pts_und_Left[indic][0][0]), int(pts_und_Left[indic][0][1])), 1, tuple(col), 2)
    
    #xy = b[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=3)
    #cv2.imwrite('fila_each{}.jpg'.format(count), left_img_remap)
    count = count + 1
    
    left_img_remap = left_img_remap2.copy()
    imagenIzq.append(ImagenL_Fila)
    ImagenL_Fila = []

plt.title('Estimated number of clusters: %d' % (n_clusters_1L + n_clusters_2L + n_clusters_3L + n_clusters_4L + n_clusters_5L))
plt.show()

np.savez('Filas_Matcheadas.npz', imagenDere=imagenDere, imagenIzq=imagenIzq)
print("Aqui",imagenDere)
contadorperron=0
for a in imagenDere:
    for b in imagenIzq:
        if a[0]==b[0]:
            left_img_remap2 = left_img_remap.copy()
            right_img_remap2 = right_img_remap.copy()
            contadorperron+=1
            print(contadorperron)
            for i in a:
                if i!=a[0]:
                    cv2.circle(right_img_remap2, (int(pts_und_Right[i][0][0]), int(pts_und_Right[i][0][1])), 1, (0,0,255), 2)
            for i in b:
                if i!=b[0]:
                    cv2.circle(left_img_remap2, (int(pts_und_Left[i][0][0]), int(pts_und_Left[i][0][1])), 1, (0,0,255), 2)
            #cv2.imwrite('winname{}.jpg'.format(contadorperron), np.hstack([left_img_remap2, right_img_remap2]))