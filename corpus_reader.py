#!/usr/bin/python

import os
import re
from itertools import islice

class AssignmentCorpus():
    def __init__(self, source_dir):
        
        self.source_dir = source_dir
        self.files = self.getFiles(self.source_dir)
        self.sents = self.readFiles(self.files, self.source_dir)
        
    def getFiles(self, source_dir):
        files = [ f for (dirpath, dirnames, filenames) in os.walk(source_dir) 
                    for f in filenames if f[-4:] == '.txt' ]

        return files
        
    def readFiles(self, files, source_dir):
        sents = []
        for fname in files:
            with open(source_dir + fname) as infile:
                for line in islice(infile.readlines(),None):
                    line = line.strip()
                    if line != '[t]':
                        for tags, sent in self.cleanLine(line):
                            sents.append((tags, sent))

        return sents
                        
    def cleanLine(self, line):
        raw_tags, sent = line.split('##')[0], line.split('##')[1:]
        tags = self.cleanTags(raw_tags)
        
        if len(sent) > 1: ## Handle exceptional lines (with more than one sentence) here
            pass 

        yield tags, sent
    
    def cleanTags(self, raw_tags):
        tags = []
        for tag in raw_tags.split(','):
                if tag != '':
                    matches = re.findall(r'(.*)\[([\+|\-])([0-9])\](?:\[(.*)\])?', tag) # with symbol r'\[([\+|\-])([0-9])\](?:\[(\w+)\])?'
                    if matches:
                        matches = matches[0]
                        text = matches[0]
                        score = int(matches[2]) if matches[1] == '+' else -int(matches[2])
                        other = matches[3]
                        tags.append((text, score, other))
                        
        return tags
