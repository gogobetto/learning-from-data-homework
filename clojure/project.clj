(defproject learning-from-data "0.1.0-SNAPSHOT"
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [incanter "1.5.6"]
                 [incanter-gorilla "0.1.0"]
                 [net.mikera/core.matrix "0.40.0"]
                 [net.mikera/vectorz-clj "0.34.0"]
                 [org.clojure/tools.nrepl "0.2.10"]
                 [org.clojure/math.numeric-tower "0.0.2"]
                 [org.clojure/math.combinatorics "0.1.1"]
                 [org.clojure/tools.trace "0.7.8"]
                 [org.clojure/data.csv "0.1.3"]
                 [clojure-csv/clojure-csv "2.0.1"]]
  :plugins [[cider/cider-nrepl "0.8.1"]]
  :profiles {:dev {:plugins [[cider/cider-nrepl "0.8.1"]]}})
