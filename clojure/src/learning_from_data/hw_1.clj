(ns learning-from-data.hw-1
  (:require [clojure.math.numeric-tower :as math]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [clojure.string :as str]
            [clojure.tools.trace :refer [deftrace]]))

;
; Math helpers

(defn cross 
    [v1 v2] 
    [ (- (* (v1 1) (v2 2)) (* (v1 2) (v2 1)))
      (- (* (v1 2) (v2 0)) (* (v1 0) (v2 2)))
      (- (* (v1 0) (v2 1)) (* (v1 1) (v2 0))) ])

(defn sgn
  [x]
  (cond (>= x 0) 1
        (< x 0) -1))

(defn avg
  [v]
  (float  (/ (reduce + v) (count v))))

(defn rand-point
  []
  (into [1] (repeatedly 2 #(- 1 (rand 2)))))

(defn make-dataset-point
  [A B]
  (let [P (rand-point)
        y (sgn ((cross (mapv - B A) (mapv - P A)) 0))]
    [P y]))

(defn make-dataset
  [N]
  (let [A (rand-point)
        B (rand-point)]
    (repeatedly N #(make-dataset-point A B))))

(defn predict
  "Given the weights, predict the classification of one point"
  [x w]
  (sgn (m/dot x w)))

(defn fraction-misclassified
  [data-set w]
  (let [mispredicted (filter
                      (fn [[P y]]
                        (not= y (predict P w)))
                      data-set)]
    (/ (count mispredicted) (count data-set))))

(defn run-pla
  "Run the PLA with a set of training points and a set of labels."
  [train-set]
  (loop [iter-count 0
         w [0. 0. 0.]]
    (let [misclassifieds (filter
                          (fn [[P y]]
                            (not= y (predict P w)))
                          train-set)]
      (if (= 0 (count misclassifieds))
        ; return a vector of [iter count, 
        [iter-count w]
        (let [ [pivot-X pivot-y] (rand-nth misclassifieds)]
          (recur (+ 1 iter-count)
                 (mapv + w (m/mul pivot-y pivot-X))))))))

(defn avg-iterations
  [N]
  (avg (mapv (fn [[iter-count w]]
               iter-count)
             (repeatedly 1000 #(run-pla
                                (make-dataset N))))))

(defn avg-eout
  [N]
  (let [complete-set (make-dataset (* N 500))]
    (avg (mapv (fn [[iter-count w]]
                 (fraction-misclassified
                  (drop N complete-set) ; leave 10 points for training
                  w))
               (repeatedly 1000 #(run-pla
                                   (take N complete-set)))))))

(defn ex-7
  []
  (avg-iterations 10))

(defn ex-8
  []
  (avg-eout 10))

(defn ex-9
  []
  (avg-iterations 100))

(defn ex-10
  []
  (avg-eout 100))
