import xml.sax
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
d = dict()
sorted_d=defaultdict(dict)
counter = 0
class GroupHandler(xml.sax.ContentHandler):
 def startElement(self,name, attrs): #Diabazei to element
    self.current = name
    #if self.current == "year":
        # print("xronos")
 def characters(self, content):

     if self.current == "year":  #An to element pou diabazei einai year pairnw to value tou
        self.year = content
        if len(self.year)==4:    #Gia na min diabazei random arithmous
         #print(self.year)
         global counter
         counter = counter + 1
         if self.year in d:       #An uparxei hdh to year auxanei to counter allios to arxikopoiei
             d[self.year] = d[self.year] + 1
         else:
             d[self.year] = 1
 def endElement(self,name):
     if self.current=='year':  #To teleutaio year pou diabazei
      if counter == 5015194:   #Autos einai o sunolikos arithmos dhmosieuseon pou brhka apo to arxeio
        for key in list(sorted(d.keys())): #Ta ektupwnei
             sorted_d[key] = d[key] #ftaixnei kaniourgio dictionary taxinomimeno
             print(key, d[key])
        nPub = (sorted_d.values())
        years = sorted(d.keys(), key = d.__getitem__,reverse = False)
        landAsx = range (86)
        plt.bar (landAsx,nPub)
        plt.xticks(landAsx,years)
        #plt.xlim([min(landAsx)-0.3,max(landAsx)+2]) #x-axonas
        plt.ylim([0, max(nPub)+3]) #o axonas y na arxizei apo to 0
        plt.grid(True, axis='y')
        plt.title('Publication per year in Dblp database',fontsize=18) #titlos
        plt.xlabel('Year(1918-2020)')
        plt.ylabel('Number of publications') #axonas - y
        plt.show()

     self.current = ""


















handler = GroupHandler()
parser = xml.sax.make_parser()
parser.setContentHandler(handler)
parser.parse(r'C:\Users\Grifrag\PycharmProjects\Giraffe\dblp-2020-04-01.xml')