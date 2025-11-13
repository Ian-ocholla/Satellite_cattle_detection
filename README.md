Over the past four decades, rising demand for livestock products in Africa has led to increased stocking rates 
resulting in overgrazing and land degradation. As the population is projected to rise, the need for sustainable 
livestock management is more urgent than ever, yet efforts are hindered by the lack of accurate, up-to-date livestock counts. 
Recent advances in remote sensing and deep learning have made it possible to count livestock from space. 
However, the extent to which models trained on aerial imagery can enhance livestock detection in satellite images
and across diverse landscapes remains limited. This study assessed the transferability of YOLO, Faster R-CNN, 
U-Net, and ResNet models for livestock detection across three contrasting landscapes, Choke bushland (Pleiades Neo), 
Kapiti savanna (WorldView-3), and LUMO open grassland (WorldView-3), using satellite imagery with 0.3 m and 0.4 m spatial resolution. 
Additionally, we applied a multi-stage transfer learning to evaluate the effectiveness of aerial imagery (0.1 m) 
trained models in improving livestock detection in satellite imagery. 

Results indicate that YOLOv5 consistently outperformed other models, achieving F1 scores of 0.55, 0.67, and 0.85 
in Choke, Kapiti, and LUMO, respectively, demonstrating robustness across varying land cover types and sensors. 
Although segmentation models performed moderately on 0.3 m imagery (F1 scores of 0.51 and 0.40 for Choke and LUMO), 
their performance dropped significantly on the coarser resolution (0.4 m) Kapiti imagery (F1 score of 0.14). 
In addition, multi-stage transfer learning improved segmentation models recall by 9.8 % in heterogeneous bushland site. 
Our results highlight that the integration of multi-source imagery and deep learning can help in large scale livestock monitoring, 
which is crucial in implementing sustainable rangeland management.
