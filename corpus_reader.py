#!/usr/bin/python

import os
import re
from itertools import islice

class AssignmentCorpus():
    def __init__(self, source_dir):
        """
        init with path do the directory with the .txt files
        """
        
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
            
            ## If you want to understand how this works, uncomment print statements
            #print 'RAW',line
            #print 
             
            # others tested, (?:(?<!\d)[\.|\?]\s+) |(?:#{2})   (?:[\.|\?]\s+)|(?:#{2})
            # best = (?:(?<!\d)[\.|\?]\s+)(?!\d.)|(?:(?<!#{2} )#{2})
            
            # split line on end of sentences AND hashes, with conditions = still not perfect but works in almost all cases
            split_line = re.split(r'(?:(?<!\d)[\.|\?]\s+)(?!\d.)|(?:(?<!#{2} )#{2})', line) 
            for i, part in enumerate(split_line):
                if re.search(r'\[([\+|\-])([0-9])\]', part): ## check if first line has any tags
                    tags =  self.cleanTags(part)
                    sent = split_line[i+1]
                    #print 'tags found'
                    #print tags
                    #print sent
                    #print
                    yield tags, sent
            
                elif i % 2 == 0 and i < len(split_line)-1: ## else yield the sentence without tags
                    tags = []
                    sent = split_line[i+1]
                    #print 'no tags'
                    #print tags
                    #print sent
                    #print
                    yield tags, sent

        elif len(sent) == 1:
            yield tags, sent[0]

        else:
            pass
    
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

if __name__ == "__main__":

    corpus = AssignmentCorpus('product_data_training_heldout/training/')
    print corpus.sents[:10]
