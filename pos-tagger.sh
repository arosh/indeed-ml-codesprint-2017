#!/bin/bash
# java -XX:TieredStopAtLevel=1 -Xverify:none -cp postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model postagger/models/english-left3words-distsim.tagger -outputFormat xml -outputFormatOptions lemmatize
java -XX:+TieredCompilation -XX:TieredStopAtLevel=1 -Xverify:none -cp postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTaggerServer -client -port 2020
