#!/bin/bash
java -XX:TieredStopAtLevel=1 -Xverify:none -cp corenlp/stanford-corenlp-3.7.0.jar edu.stanford.nlp.process.DocumentPreprocessor
