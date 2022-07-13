## This is the repo for Day 9 - Intent Detection and Slot Filling.

#### Task to perform: 
- <b>Input Sentence :</b>  does flight dl 1083 from philadelphia to denver fly on saturdays
- <b>Truth        :</b>  O O B-airline_code B-flight_number O B-fromloc.city_name O B-toloc.city_name O O B-depart_date.day_name
- <b>Prediction :</b>  O O B-airline_code I-airline_name O B-fromloc.city_name O B-toloc.city_name O O B-depart_date.day_name
- <b>Truth        :</b>  atis_flight
- <b>Prediction :</b>  atis_flight
