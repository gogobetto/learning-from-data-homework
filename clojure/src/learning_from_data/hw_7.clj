(ns learning-from-data.hw-7
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

(defn to-feature-space
  "Map a [x0 x1 x2] point to feature space."
  [[[x0 x1 x2] y] degree]
  [(take (inc degree) [x0
                    x1
                    x2
                    (* x1 x1)
                    (* x2 x2)
                    (* x1 x2)
                    (Math/abs (- x1 x2))
                    (Math/abs (+ x1 x2))])
   y])

(defn fraction-misclassified
  "Takes a [[x0 x1 x2] y] set and the computed weights. Returns the fraction of misclassified points"
  [set w]
  (let [my-class (mapv #(sgn (dot (first %) w)) set)
        is-miscl? #(if (= (last %1) %2) 0 1) ; apply to dataset point and computed y
        count-mis (reduce + (mapv is-miscl? set my-class))]
    (float (/ count-mis (count set)))))

(defn do-regression
  [trainset
   testset]
  (let [X (mapv first trainset)
        X_cross (pseudoinverse X 0)
        y (mapv last trainset)
        w (m/mmul X_cross y)]
    [ (fraction-misclassified trainset w)
         (fraction-misclassified testset w)]))

(defn ex1
  []
  (doseq [k [3 4 5 6 7]]  
    (println "k" k (mapv #(format)) (do-regression (->> (read-file "in.dta")
                                                  (mapv #(to-feature-space % k))
                                                  (take 25))
                                     (->> (read-file "in.dta")
                                          (mapv #(to-feature-space % k))
                                          (drop 25)
                                          (take 10))))))

(defn ex2
  []
  (doseq [k [3 4 5 6 7]]
    (println "k" k (do-regression (->> (read-file "in.dta")
                                       (mapv #(to-feature-space % k))
                                       (take 25))
                                  (->> (read-file "out.dta")
                                       (mapv #(to-feature-space % k)))))))

(defn ex3
  []
  (doseq [k [3 4 5 6 7]]  
    (println "k" k (do-regression (->> (read-file "in.dta")
                                       (mapv #(to-feature-space % k))
                                       (drop 25)
                                       (take 10))
                                  (->> (read-file "in.dta")
                                       (mapv #(to-feature-space % k))
                                       (drop 0)
                                       (take 25))))))

(defn ex4
  []
  (doseq [k [3 4 5 6 7]]  
    (println "k" k (do-regression (->> (read-file "in.dta")
                                       (mapv #(to-feature-space % k))
                                       (drop 25)
                                       (take 10))
                                  (->> (read-file "out.dta")
                                       (mapv #(to-feature-space % k)))))))

(defn ex5
  "Out of sample performance for k=6 models in ex1 and ex3"
  []
  (let [k 6]
    (println "k" k (do-regression (->> (read-file "in.dta")
                                       (mapv #(to-feature-space % k))
                                       (take 25))
                                     (->> (read-file "out.dta")
                                          (mapv #(to-feature-space % k)))))
    (println "k" k (do-regression (->> (read-file "in.dta")
                                       (mapv #(to-feature-space % k))
                                       (drop 25)
                                       (take 10))
                                  (->> (read-file "out.dta")
                                       (mapv #(to-feature-space % k)))))))

(defn ex6
  []
  (let [run-exp (fn []
                  (let [e1 (rand 1)
                        e2 (rand 1)]
                    (min e1 e2)))
        exp-cnt 100]
    (/
     (reduce + (repeatedly exp-cnt run-exp))
     exp-cnt)))

;
; PLA code

