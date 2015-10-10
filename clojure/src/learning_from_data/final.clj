(ns learning-from-data.final
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
    straight))

(defn read-file
  [file-name]
  (with-open [rd (io/reader (io/file file-name))]
    (doall
     (mapv parse-line (line-seq rd)))))

(defn make-vs-all-dataset
  [data n]
  (mapv (fn [row]
          (if (= n  (int  (row 0)))
            [(into [1.] (drop 1 row)) 1.]
            [(into [1.] (drop 1 row)) -1.]))
        data))

(defn make-one-vs-other-dataset
  [data one other]
  (filter identity ; filter out nil elements
          (mapv (fn [row]
                  (cond
                   (= one (int (row 0)))
                   [(into [1.] (drop 1 row)) 1.]
                   (= other (int (row 0)))
                   [(into [1.] (drop 1 row)) -1.]
                   :else
                   nil)) data)))

(defn fraction-misclassified
  "Takes a [[x0 x1 x2] y] vector and the computed weights. Returns the fraction of misclassified points"
  [set w]
  (let [my-class (mapv #(sgn (dot (first %) w)) set)
        is-miscl? #(if (= (last %1) %2) 0 1) ; apply to dataset point and computed y
        count-mis (reduce + (mapv is-miscl? set my-class))]
    (float (/ count-mis (count set)))))

(defn do-regression
  [train-set test-set lambda]
  (let [X (mapv first train-set)
        X_cross (pseudoinverse X lambda)
        y (mapv last train-set)
        w (m/mmul X_cross y)]
    [ (fraction-misclassified train-set w)
      (fraction-misclassified test-set w)]))

(defn ex7
  []
  (let [all-train (read-file "features.train")
        all-test (read-file "features.test")] 
    (map (fn [digit] (do-regression
                       (make-vs-all-dataset all-train digit)
                       (make-vs-all-dataset all-test digit)
                       1)) [5 6 7 8 9])))

; (ex7)

(defn feat-transform
  [x]
  (let [x1 (x 1)
        x2 (x 2)]
    [1 x1 x2 (* x1 x2) (* x1 x1) (* x2 x2)]))

(defn transform-dataset
  [data]
  (mapv (fn [p]
          [(feat-transform (p 0))
           (p 1)]) data ))

(defn ex8
  []
  (let [all-train (read-file "features.train")
        all-test (read-file "features.test")] 
    (map (fn [digit] (do-regression
                      (transform-dataset
                       (make-vs-all-dataset all-train digit))
                      (transform-dataset
                       (make-vs-all-dataset all-test digit))
                      1)) [0 1 2 3 4])))


; (ex8)

(defn ex9
  []
  (let [all-train (read-file "features.train")
        all-test (read-file "features.test")]
    (map (fn [digit]
           (let [digit-train (make-vs-all-dataset all-train digit)
                 digit-test (make-vs-all-dataset all-test digit)]
             {:without (do-regression
                        digit-train
                        digit-test
                        1)
              :with (do-regression
                     (transform-dataset digit-train)
                     (transform-dataset digit-test)
                     1)}))
         (range 10))))

; (ex9)

(defn ex10
  []
  (let [train-set (transform-dataset
                   (make-one-vs-other-dataset
                    (read-file "features.train")
                    1 5))
        test-set (transform-dataset
                  (make-one-vs-other-dataset
                   (read-file "features.test")
                   1 5))]
    (map (fn [lambda]
           (do-regression train-set
                          test-set
                          lambda))
         [1 0.01])))

; (ex10)
