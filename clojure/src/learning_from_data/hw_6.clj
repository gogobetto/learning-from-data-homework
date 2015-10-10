(ns learning-from-data.hw-6
  (:require [clojure.math.numeric-tower :as math]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [clojure.string :as str]
            [clojure.tools.trace :refer [deftrace]]))

;
; Math utility functions

(defn pseudoinverse
  [X lambda]
  (let [X_t (m/transpose X)
        dim (count (first X))
        decay-mat (m/mmul lambda (m/identity-matrix dim))]
    (m/mmul (m/inverse (m/add (m/mmul X_t X) decay-mat)) X_t)))

(defn square
  [x]
  (* x x))

(defn sgn
  [x]
  (if (>= x 0)
    1.0
    -1.0))

(defn dot
  "Dot product of two vectors."
  [u v]
  ;(println "dot: " u v)
  (reduce + (map * u v)))

;
;

(defn parse-line
  "Parse a string line and return a [[x0 x1 x2] y] vector."
  [l]
  (let [straight 
        (->> (str/split l #"\s+")
             (filter #(not (str/blank? %)))
             (mapv #(Float/parseFloat %)))]
    [(into [1] (take 2 straight)) (nth straight 2)]))

(defn read-file
  [file-name]
  (with-open [rd (io/reader (io/file file-name))]
    (doall
     (mapv parse-line (line-seq rd)))))

;
; Exercise 1

(defn to-feature-space
  "Map a [x0 x1 x2] point to feature space."
  [[[x0 x1 x2] y]]
  [[x0 x1 x2 (* x1 x1) (* x2 x2) (* x1 x2) (Math/abs (- x1 x2)) (Math/abs (+ x1 x2))] y])

(defn fraction-misclassified
  "Takes a [[x0 x1 x2] y] set and the computed weights. Returns the fraction of misclassified points"
  [set w]
  (let [my-class (mapv #(sgn (dot (first %) w)) set)
        is-miscl? #(if (= (last %1) %2) 0 1) ; apply to dataset point and computed y
        count-mis (reduce + (mapv is-miscl? set my-class))]
    (float (/ count-mis (count set)))))

(defn do-regression
  [lambda]
  (let [trainset (->> (read-file "in.dta")
                      (mapv to-feature-space))
        X (mapv first trainset)
        X_cross (pseudoinverse X lambda)
        y (mapv last trainset)
        w (m/mmul X_cross y)
        testset (->> (read-file "out.dta")
                     (mapv to-feature-space))]
    [ (fraction-misclassified trainset w)
      (fraction-misclassified testset w)]))

; (println (read-file "in.dta"))
(println "Lambda=0" (do-regression 0))
(println "Lambda=0.001" (do-regression 0.001))
(println "Lambda=1000" (do-regression 1000))

(println "****")

(doseq [k [2 1 0 -1 -2]]
  (let [lambda (Math/pow 10 k)]
    (println "Lambda=" lambda (do-regression lambda))))
