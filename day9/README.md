## This is the repo for Day 9 - Intent Detection and Slot Filling.

#### Task to perform: 
- <b>Input Sentence :</b>  does flight dl 1083 from philadelphia to denver fly on saturdays
- <b>Truth        :</b>  O O B-airline_code B-flight_number O B-fromloc.city_name O B-toloc.city_name O O B-depart_date.day_name
- <b>Prediction :</b>  O O B-airline_code I-airline_name O B-fromloc.city_name O B-toloc.city_name O O B-depart_date.day_name
- <b>Truth        :</b>  atis_flight
- <b>Prediction :</b>  atis_flight

-----
#### References:
1. [Github - RNN-for-Joint-NLU](https://github.com/DSKSD/RNN-for-Joint-NLU)
2. Liu, Bing, and Ian Lane. "Attention-based recurrent neural network models for joint intent detection and slot filling." arXiv preprint arXiv:1609.01454 (2016). [Paper](https://arxiv.org/pdf/1609.01454.pdf)
3. [Dataset](https://github.com/yvchen/JointSLU/tree/master/data)
4. [Intent Classification with SVM](https://www.kaggle.com/code/oleksandrarsentiev/intent-classification-with-svm)
5. [Airline-Travel-Information-System-ATIS-Text-Analysis](https://github.com/nawaz-kmr/Airline-Travel-Information-System-ATIS-Text-Analysis#airline-travel-information-system-atis-text-analysis)
