(ns learning-from-data.hw-2
  (:require [clojure.math.numeric-tower :as math]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [clojure.string :as str]
            [clojure.tools.trace :refer [deftrace]]))

(m/set-current-implementation :vectorz)

(defn cross 
    [v1 v2] 
    [ (- (* (v1 1) (v2 2)) (* (v1 2) (v2 1)))
      (- (* (v1 2) (v2 0)) (* (v1 0) (v2 2)))
      (- (* (v1 0) (v2 1)) (* (v1 1) (v2 0))) ])

(defn sgn
  [x]
  (cond (> x 0) 1
        (< x 0) -1
        (= x 0) 1))

(defn avg [x]
  (float (/ (reduce + x) (count x))))

;; @@
;
; QUESTIONS 1 and 2

(def flips-num 10)
(def coins-num 1000)

; HEAD = 1
; TAIL = 0

(defn run-trial
  [_]
  (let [flip-results (mapv (fn [_] (map (fn [_] (rand-int 2)) (range 0 flips-num))) (range 0 coins-num))
        c_1 (first flip-results)
        c_rand (rand-nth flip-results)
        c_min (apply min-key #(sum %1) flip-results)]
	(mapv (comp #(/ %1 flips-num) sum) [c_1 c_rand c_min])))


(def trials-num 1)
(def results (map run-trial (range trials-num)))
; zip the results and build the averages

(map avg (apply map vector results))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;tasteless-waterfall/results</span>","value":"#'tasteless-waterfall/results"}
;; <=

;; @@
;
; QUESTION 3

; Let's build a probability tree. There are two (independent) events for which the approximation is
; not correct:
; h=f and f!=y and h!=f and f=y
; We just add their probability since they are independent

;
; QUESTION 4

; We refactor question 3 as: 1-L+U(2L-1) and get rid of the dependency on U, so L=1/2
;; @@

;; @@
;
; QUESTION 5, 6, 7

(defn rand-point []
  ; Add 1 as x_0..
  (into [1] (take 2 (repeatedly #(- 1 (rand 2))))))

(defn run-f
  "Which side of the AB segment is point P on? (cross product)"
  [A B P]
  (sgn ((cross (mapv - B A) (mapv - P A)) 0)))

(defn make-dataset-point
  [A B]
  (let [P (rand-point)]
    [P (run-f A B P)]))

(defn pseudoinverse
  [X]
  (let [X_t (m/transpose X)]
    (m/mmul (m/inverse (m/mmul X_t X)) X_t)))

(defn fraction-misclassified
  "Given a set of points, their `right` y and the weights (g),
   compute the fraction of misclassified ones."
  [X y w]
	(let [my_class (map #(sgn (m/dot %1 w)) X)
          count_mis (reduce + (mapv #(if (= %1 %2) 0 1) my_class y))]
      (float (/ count_mis (count y)))))

(defn run-pla
  "Run the PLA with a set of training points, their right y,
   and the initial weights"
  [X y init_w]
  (loop [iter-count 0
         w 			init_w]
    (let [my_y					(mapv #(sgn (m/dot %1 w)) X) ; predicted y with the current w
          tmp					(mapv vector X y my_y) ; temporary data structure to filter
          misclassifieds		(filter #(not (= (%1 1) (%1 2))) tmp)]
      (if (= 0 (count misclassifieds))
        iter-count
        (let [pivot (rand-nth misclassifieds)]
          ;(println (str "pivoting" pivot))
          (recur (+ 1 iter-count)
                 (mapv + w (m/mul (pivot 1) (pivot 0)))))))))


         
         
(defn ex5-regression
  "Runs the regressions and returns the fraction of misclassified points.
   Takes as inputs the N number of items that should be in the training set."
  [N] ; number of points in the training dataset
  (let [A (rand-point)
        B (rand-point)
        trainset (into [] (take N (repeatedly #(make-dataset-point A B))))
        X (mapv #(first %1) trainset)
        X_cross (pseudoinverse X)
        y (mapv #(last %1) trainset)
        w (m/mmul X_cross y)
        ; validation dataset with 1000 fresh points
        valset (into [] (take 1000 (repeatedly #(make-dataset-point A B))))]
    
    ; Returns a vector with:
    ; weights
    ; E_in
    ; E_out
    [ w
      (fraction-misclassified X y w)
      (fraction-misclassified (mapv #(first %1) valset)
                              (mapv #(last %1) valset)
                              w)
      (run-pla X y w)]
    ;(println w)
    ;(println (map #(m/dot %1 w) X))
    ; (chart-view (charts/scatter-plot (map #(%1 1) X) (map #(%1 2) X)))
))

; Run the experiment 1000 times with N=100 and take the average
; Returns a vector [avg E_in, avg E_out, avg PLA iterations]
(map avg (apply map vector (map rest (into [] (repeatedly 1000 #(ex5-regression 10))))))

;; @@
;
; Exercise 8

(defn run-f-circle
  [[x0 x1 x2]]
  (sgn (+ (* x1 x1) (* x2 x2) -0.6)))

(defn make-noisy-dataset-point
  "Flip the y 10% of the time"
  []
  (let [P (rand-point)]
    (if (= 8 (rand-int 10))
      [P (- (run-f-circle P))]
      [P (run-f-circle P)])))


(defn ex8-regression
  [N] ; number of points in the training dataset
  (let [trainset (into [] (repeatedly N make-noisy-dataset-point))
        X (mapv #(first %1) trainset)
        X_cross (pseudoinverse X)
        y (mapv #(last %1) trainset)
        w (m/mmul X_cross y)]
    
    (fraction-misclassified X y w)
))

(repeatedly 1000 #(ex8-regression 1000))

;; @@
;
; Exercises 9 and 10

(defn to-feature-space
  [[[x0 x1 x2] y]]
  [[x0 x1 x2 (* x1 x2) (* x1 x1) (* x2 x2)] y])

(defn ex9-regression
  [N] ; number of points in the training dataset
  (let [trainset (into [] (repeatedly N #(to-feature-space (make-noisy-dataset-point))))
        X (mapv #(first %1) trainset)
        X_cross (pseudoinverse X)
        y (mapv #(last %1) trainset)
        w (m/mmul X_cross y)
        ; validation dataset (1000 fresh points)
        valset (into [] (repeatedly 1000 #(to-feature-space (make-noisy-dataset-point))))]
    
    [w
     (fraction-misclassified (mapv #(first %1) valset)
                              (mapv #(last %1) valset)
                              w)]
))

(def results (repeatedly 1000 #(ex9-regression 1000)))
; ex 9: weights vector
(map avg (apply map vector (map first results)))
; ex 10: E_out
(avg (map last results))
