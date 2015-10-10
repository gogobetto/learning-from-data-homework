(ns learning-from-data.hw-5
  (:require [clojure.math.numeric-tower :as math]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [clojure.string :as str]
            [clojure.tools.trace :refer [deftrace]]))

(use 'clojure.tools.trace)

;
; Math helpers
;

(defn square
  [x]
  (Math/pow x 2))

(defn cross
  "Cross product of two R3 vectors."
  [v1 v2]
  [ (- (* (v1 1) (v2 2)) (* (v1 2) (v2 1)))
    (- (* (v1 2) (v2 0)) (* (v1 0) (v2 2)))
    (- (* (v1 0) (v2 1)) (* (v1 1) (v2 0))) ])

(defn dot
  "Dot product of two vectors."
  [u v]
  (reduce + (map * u v)))

(defn sgn
  "Sign function."
  [x]
  (cond (> x 0) 1
        (< x 0) -1
        (= x 0) 1))

(defn avg [x]
  "Average of a vector."
  (float (/ (reduce + x) (count x))))

(defn norm
  "Euclidean norm of a vector."
  [v]
  (Math/sqrt
   (reduce + (map #(* % %) v))))

;
; Exercises 5-6-7

(defn E
  [u v]
  (square (- (* u (Math/exp v)) (* 2 v (Math/exp (- u))))))

(defn dEdu
  [u v]
  (* 2 (+ (Math/exp v) (* 2 v (Math/exp (- u)))) (- (* u (Math/exp v)) (* 2 v (Math/exp (- u))))))
  
(defn dEdv
  [u v]
  (* 2 (- (* u (Math/exp v)) (* 2 v (Math/exp (- u)))) (- (* u (Math/exp v)) (* 2 (Math/exp (- u))))))

(defn run-gradient-descent
  [u-init v-init tolerance]
  (let [ni 0.1]  
    (loop [iter-count	0
           u			u-init
           v			v-init]
      ;(println u v (E u v))
      (if (or (< (E u v) tolerance) (> iter-count 1000))
        iter-count
      	(let [new-u	(- u (* ni (dEdu u v)))
              new-v	(- v (* ni (dEdv u v)))]
          (recur (inc iter-count)
                 new-u
                 new-v))))))

(println "Gradient descent" 
         (run-gradient-descent 1 1 1E-14))

(defn run-coordinate-descent
  [u-init v-init iter-stop]
  (let [ni 0.1]  
    (loop [iter-count	0
           u			u-init
           v			v-init]
      ;(println u v (E u v))
      (if (> iter-count iter-stop)
        [u v (E u v)]
        (let [new-u	(- u (* ni (dEdu u v)))
              new-v	(- v (* ni (dEdv new-u v)))]
          (recur (inc iter-count)
                 new-u
                 new-v))))))

(println "Coordinate descent" 
         (run-coordinate-descent 1 1 15))

;
; Exercises 8-9

(defn rand-point []
  ; Add 1 as x_0..
  (into [1] (take 2 (repeatedly #(- 1 (rand 2))))))

(defn run-f
  "Which side of the AB segment is point P on? (sign of cross product)"
  [A B P]
  (sgn ((cross (mapv - B A) (mapv - P A)) 0)))

(defn make-dataset-point
  [A B]
  (let [P (rand-point)]
    [P (run-f A B P)]))

(defn make-dataset
  [n A B]
  (into [] (repeatedly n #(make-dataset-point A B))))

;
;

(defn logistic-regr-gradient
  [w x y]
;  (println (str "gradient " y))
;  (println (str "gradient " w " " x " " y))
  (let [f (/ y (+ 1 (Math/exp (* y (dot w x)))))]
    (mapv #(* f %) x)))

(defn do-epoch
    "Run an epoch and return the final weights."
    [train-set
     start-w]
    (let [ni 0.01] 
      (loop [train       (shuffle train-set)
             one-over-N  (/ 1 (count train-set)) ; normalization
             w	       start-w]
        (if (empty? train)
          w
          (let [sample	(first train)
                grad	(logistic-regr-gradient
                         w
                         (first sample) ; x vector
                         (last sample)) ; y real
                step	(map #(* ni %) grad)]
                                        ; (println w)
            (recur (rest train)
                   one-over-N
                   (map + step w)))))))

(defn cross-entropy-error
  "Given A, B, and weights, compute the CE error on a new dataset."
  [A B w]
  (let [cnt 100
        valset (make-dataset cnt A B)
        error-fn (fn [sample]
                   (Math/log
                    (+ 1 (Math/exp
                          (- (* (last sample)
                                (dot w (first sample))))))))]
    (avg (map error-fn valset))))

(defn run-sgd
  [N]
  (let [A 			(rand-point)
        B 			(rand-point)
        train-set 	(make-dataset N A B)]
    (loop [i-epoch	0
           w_t		[0 0 0]
           w_t-1	[1 1 1]]
;      (println (pr-str "epoch " i-epoch  w_t))
      (if (or (< (norm (map - w_t w_t-1)) 0.01) (> i-epoch 1000))
        [i-epoch (cross-entropy-error A B w_t)]
        (recur (inc i-epoch)
               (do-epoch train-set w_t)
               w_t)))))

(defn ex8-9
  []
  (let [results (repeatedly 100 #(run-sgd 100))]
    (println "SGD, Avg epochs" (avg (map first results)))
    (println "SGD, Avg E_out" (avg (map last results)))))

(ex8-9)
