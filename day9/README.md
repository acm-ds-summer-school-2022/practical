This is the repo for Day 9 - Intent Detection and Slot Filling.



Task to perform: 
- Input Sentence :  does flight dl 1083 from philadelphia to denver fly on saturdays
- Truth        :  O O B-airline_code B-flight_number O B-fromloc.city_name O B-toloc.city_name O O B-depart_date.day_name
- Prediction :  O O B-airline_code I-airline_name O B-fromloc.city_name O B-toloc.city_name O O B-depart_date.day_name
- Truth        :  atis_flight
- Prediction :  atis_flight
