#!/bin/bash
java -mx1g -XX:TieredStopAtLevel=1 -Xverify:none -cp postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTaggerServer -model postagger/models/english-left3words-distsim.tagger -outputFormat xml -outputFormatOptions lemmatize -port 2020
