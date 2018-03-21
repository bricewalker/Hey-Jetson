# werpp.py: Calculates WER and paints the edition operations
# Copyright (C) 2017 Nicolás Serrano Martínez-Santos <nserrano@dsic.upv.es>
# Contributors: Guillem Gasco, Adria A. Martinez, Daniel Martín-Albo


import codecs
import re
from random import shuffle
from sys import argv,stderr,stdout,maxsize
from optparse import OptionParser
import codecs
import array
from copy import copy

class FileReader:
  def __init__(self,f,buffer_size=1024):
    #open file
    self.f = f
    self.buffer_size = 1024
    self.buff_readed = 0
    self.buff_len = 0

  def read_buff(self):
    self.buff = self.f.read(self.buffer_size)

    self.buff_len = len(self.buff)
    self.buff_readed = 0

    if self.buff_len == 0:
      return False
    else:
      return True

  def readline(self):
    s = ""
    while 1:
      while self.buff_readed < self.buff_len:
        if self.buff[self.buff_readed] == '\n':
          self.buff_readed+=1
          return s
        else:
          s+=self.buff[self.buff_readed]
          self.buff_readed+=1

      if not self.read_buff():
        return None

  def close(self):
    self.f.close()

#awk style dictionary
class D(dict):
  def __getitem__(self, i):
    if i not in self: self[i] = 0
    return dict.__getitem__(self, i)

#awk style dictionary
class Dincr():
  def __init__(this):
    this.n = 0
    this.anti_d = {}
    this.d = {}
  def dic(self, i):
    if i not in self.d:
      self.d[i] = self.n
      self.anti_d[self.n] = i
      self.n += 1
    return self.d[i]
  def inv(self, i):
    return self.anti_d[i]

class color:
  d={}; RESET_SEQ=""
  def __init__(self,c):
    if c == True:
      self.d['K']="\033[0;30m"    # black
      self.d['R']="\033[0;31m"    # red
      self.d['G']="\033[0;32m"    # green
      self.d['Y']="\033[0;33m"    # yellow
      self.d['B']="\033[0;34m"    # blue
      self.d['M']="\033[0;35m"    # magenta
      self.d['C']="\033[0;36m"    # cyan
      self.d['W']="\033[0;37m"    # white
      self.RESET_SEQ = "\033[0m"
    else:
      self.d['K']="["
      self.d['R']="<"
      self.d['G']="["
      self.d['Y']="["
      self.d['B']="["
      self.d['M']="["
      self.d['C']="["
      self.d['W']="["
      self.RESET_SEQ = "]"

  def c_string(self,color,string):
    return self.d[color]+string+self.RESET_SEQ

# Normal compare strings
def string_equal(str1,str2):
  return (str1 == str2)

# Ignore simbol #
def dummy_string_equal(str1,str2):
  return (str1.replace("#","") == str2)

def string_equal_lowercase(str1,str2):
  return (str1.lower() == str2.lower())

#read lines and return its simbol representation
def char_to_num(x):
  res =""
  s=x
  for i in s:
    if i == " ":
      res += "__ "
    else:
      res += "%d " %ord(i)
  return res[:-1]

def num_to_char(j):
  if j != "__":
    return unichr(int(j))
  else:
    return j

class e_op:
  def __init__(this,ins,dels,subs):
    this.i = ins; this.d = dels; this.s = subs
  def cost(this):
    return this.i + this.d + this.s
  def ins(this):
    return this.i
  def dels(this):
    return this.d
  def subs(this):
    return this.s
  def __repr__(this):
    return "I:%d D:%d S:%d" %(this.i,this.d,this.s)

#only computes the cost of the best path    
def lev_changes_naive(str1, str2, eq_func=string_equal):

  d_prev=[]
  d_curr=[]

  for i in xrange(len(str1)+1):
    d_prev.append(e_op(0,i,0))
    d_curr.append(e_op(0,0,0))

  for i in xrange(1, len(str2)+1):
    for j in xrange(1, len(str1)+1):
      if j == 1:
        d_curr[0].i = i

      equals = (1- eq_func(str1[j-1],str2[i-1]))
      sub_cost = d_prev[j-1].cost() + equals
      ins_cost = d_prev[j].cost() + 1
      del_cost = d_curr[j-1].cost() + 1

      if sub_cost < ins_cost and sub_cost < del_cost:
        d_curr[j].i = d_prev[j-1].i
        d_curr[j].d = d_prev[j-1].d
        d_curr[j].s = d_prev[j-1].s + equals
      elif ins_cost < del_cost:
        d_curr[j].i = d_prev[j].i + 1
        d_curr[j].d = d_prev[j].d
        d_curr[j].s = d_prev[j].s
      else:
        d_curr[j].i = d_curr[j-1].i
        d_curr[j].d = d_curr[j-1].d + 1
        d_curr[j].s = d_curr[j-1].s

    aux = d_prev
    d_prev = d_curr
    d_curr = aux

  return d_prev[len(str1)].ins(), d_prev[len(str1)].dels(), d_prev[len(str1)].subs()


def lev_changes(str1, str2, i_cost, d_cost, d_sub,vocab={}, eq_func=string_equal):
  d={}; sub={};
  for i in range(len(str1)+1):
    d[i]=dict()
    d[i][0]=i*d_cost
    sub[i]={}
    sub[i][0]='D'
  for i in range(len(str2)+1):
    d[0][i] = i*i_cost
    sub[0][i]='I'
  for i in range(1, len(str1)+1):
    for j in range(1, len(str2)+1):
      cur_i_cost = d[i][j-1] + i_cost
      cur_d_cost = d[i-1][j] + d_cost
      cur_sub_cost = d[i-1][j-1] + (not eq_func(str1[i-1],str2[j-1]))*d_sub

      #Calculate min cost
      d[i][j] = min(cur_sub_cost, cur_i_cost, cur_d_cost)

      #Store the path to retrieve the operation
      if cur_sub_cost < cur_i_cost and cur_sub_cost < cur_d_cost:
        if eq_func(str1[i-1],str2[j-1]):
          sub[i][j] = 'E';
        else:
          if vocab=={} or (str2[j-1] in vocab):
            sub[i][j] = 'S';
          else:
            sub[i][j] = 'A'; #Oov Substitution
      elif cur_i_cost < cur_d_cost:
        if vocab=={} or (str2[j-1] in vocab):
          sub[i][j] = 'I';
        else:
          sub[i][j] = 'O'; #Oov insertion
      else:
        sub[i][j] = 'D';


  i=len(str1); j=len(str2); path=[]
  while(i > 0 or j > 0):
    path.append([sub[i][j],i-1,j-1])
    if(sub[i][j] == 'I' or sub[i][j] == 'O'):
      j-=1
    elif(sub[i][j] == 'D'):
      i-=1
    else:
      j-=1; i-=1;
  path.reverse()
  return path

def calculate_statistics(rec_file, ref_file, options):
  subs={}; subs_counts=D(); subs_all = 0
  ins=D(); ins_all = 0
  dels=D(); dels_all = 0

  #There is an external vocab for OOV
  words=Dincr(); n_words = 0
  vocab = {}
  if options.vocab != None:
    f = codecs.open(options.vocab,"r","utf-8")
    for i in f.readlines():
      for j in i.split():
        if words.dic(j) not in vocab:
          vocab[words.dic(j)]=1

  join_symbol="@"
  colors=color(options.color)
  oovSubs=0
  oovIns=0
  oovs = 0
  ref_count=0

  n_pressed_keys = 0

  eq_func = string_equal

  #change compare function
  if options.equal_func == "dummy":
    eq_func = dummy_string_equal
  elif options.equal_func == "lower":
    eq_func = string_equal_lowercase

  excps = []
  if options.excp_file != None:
    f = codecs.open(options.excp_file,"r","utf-8")
    for i in f.readlines():
      excps.append(i[:-1])
    f.close()

  if options.v == True:
    stdout.write(colors.RESET_SEQ)

  i = rec_file.readline()
  while len(i) != 0:
    j = ref_file.readline()

    if options.cer:
      if options.equal_func == "lower":
        i = char_to_num(i[:-1].lower())
        j = char_to_num(j[:-1].lower())
      else:
        i = char_to_num(i[:-1])
        j = char_to_num(j[:-1])

    words_i = i.split()
    words_j = j.split()



    w_i = []
    for i in words_i:
      #delete some words if necessary
      if options.excp_file != None:
        if i not in excps:
          w_i.append(words.dic(i))
      else:
        w_i.append(words.dic(i))

    w_j = []
    for i in words_j:
      if options.excp_file != None:
        if i not in excps:
          w_j.append(words.dic(i))
      else:
        w_j.append(words.dic(i))

    if len(w_j) == 0:
      if options.ignore_blank == True:
        stderr.write("[WW] Blank line in reference, ignoring it\n")
        i = rec_file.readline()
        continue
      else:
        stderr.write("[WW] Blank line in reference\n")
        if options.v == True:
          stdout.write("[II] ")
          for index in w_i:
            if options.cer:
              rec = num_to_char(index)
            else:
              rec = words.inv(index)
            str_out = "%s" %(colors.c_string("R",rec).encode("utf-8"))
            if not options.cer:
              str_out = str_out+" "
            elif "__" in str_out:
              str_out = " "+str_out+" "
            stdout.write(str_out)
            dels_all+=1
            dels[rec]+=1
          stdout.write("\n")
        else:
          dels_all+=len(w_i)
      i = rec_file.readline()
      continue

    ref_count+= len(w_j)

    if options.v == None and options.n == 0 and options.color == None and \
          options.vocab == None and options.key_pressed == None:
      ins_naive = del_naive = subs_naive = 0
      if len(w_i) == 0:
        ins_naive += len(w_j)
      else:
        ins_naive, del_naive, subs_naive = lev_changes_naive(w_i,w_j)
      ins_all += ins_naive; dels_all += del_naive; subs_all += subs_naive
      v_editions = ins_naive + del_naive + subs_naive
    else:
      changes = lev_changes(w_i,w_j,1,1,1,vocab,eq_func)

      if options.v == True:
        stdout.write("[II] ")
      #verbose variables
      v_editions=0

      for i in changes:
        [edition, rec_p, ref_p] = i
        rec = words.inv(w_i[rec_p]) if len(w_i) > 0 else "#"
        ref = words.inv(w_j[ref_p])
        if options.cer:
          rec = num_to_char(rec)
          ref = num_to_char(ref)

        #color the operations
        if options.v == True:
          str_out = ""
          if edition == 'S':
            str_out = "%s" %(colors.c_string("B",rec+join_symbol+ref).encode("utf-8"))
          elif edition == 'A':
            str_out = "%s" %(colors.c_string("Y",rec+join_symbol+ref).encode("utf-8"))
          elif edition == 'I':
            str_out = "%s" %(colors.c_string("G",ref).encode("utf-8"))
          elif edition == 'D':
            str_out = "%s" %(colors.c_string("R",rec).encode("utf-8"))
          elif edition == 'O':
            str_out = "%s" %(colors.c_string("Y",ref).encode("utf-8"))
          else:
            str_out = "%s" %ref.encode("utf-8")
          if not options.cer:
            str_out = str_out+" "
          elif "__" in str_out:
            str_out = " "+str_out+" "
          stdout.write(str_out)


        #count the segment where the errors occur
        if edition != 'E':
          #WER on each line
          if options.V == 1:
            v_editions+=1
          if options.vocab != None:
            if ref not in vocab:
              oovs+=1

        #count events in dictionaries
        if edition == 'S' or edition == 'A':
          subs_all+=1
          if edition == 'A':
            oovSubs+=1
          if options.n > 0:
            if ref not in subs:
              subs[ref]={}
            if rec not in subs[ref]:
              subs[ref][rec] = 1
            else:
              subs[ref][rec]+=1
          subs_counts[ref]+=1

        elif edition == 'I' or edition == 'O':
          if edition == 'O':
            oovIns+=1
          ins_all+=1
          ins[ref]+=1
        elif edition == 'D':
          dels_all+=1
          dels[rec]+=1

        #count number of pressed keys Eq == 1 else Ref
        if edition == 'E':
          n_pressed_keys += 1
        elif edition == 'D':
          n_pressed_keys += 1
        else:
          n_pressed_keys += len(ref)+1

      if options.v == True:
        stdout.write("\n")

    if options.V == 1:
      stdout.write("[II] WER-per-sentence Eds: %d Ref: %d\n" %(v_editions,len(w_j)))

    i = rec_file.readline()

  if ref_count == 0:
    stderr.write("[EE] There are not words in the reference. WER will not be calculated.\n")
    exit(1)
  else:
    if options.cer:
      tag="CER"
    else:
      tag="WER"
    stdout.write("%s: %.2f (Ins: %d Dels: %d Subs: %d Ref: %d )" \
        %(tag, float(subs_all+ins_all+dels_all)/ref_count*100,ins_all,dels_all,subs_all,ref_count))

  if options.vocab != None:
   # stdout.write(" OOVs: %.2f%%" %(float(oovs)/ref_count*100))
    stdout.write(" OOVs: %.2f%%" %(float(oovSubs+oovIns)/ref_count*100))
    stdout.write(" OOVsSubs: %.2f%%" %(float(oovSubs)/subs_all*100))
    stdout.write(" OOVsIns: %.2f%%" %(float(oovIns)/ins_all*100))
  stdout.write("\n")

  if options.key_pressed:
    stdout.write("Number of pressed keys: %d\n" %n_pressed_keys)

  if options.n > 0:
    stdout.write("----------------------------------\nWer due to words words\n----------------------------------\n")
    events=[]
    for i in subs:
      for j in subs[i]:
        events.append([subs[i][j],['S',i,j]])
    for i in ins:
      events.append([ins[i],['I',i]])
    for i in dels:
      events.append([dels[i],['D',i]])

    events.sort(); acc=0
    for i in range(len(events)-1,len(events)-1-options.n,-1):
      [n, e] = events[i]
      s=""
      if 'S' in e:
        s=colors.c_string("B",e[2]+join_symbol+e[1])
      elif 'I' in e:
        s=colors.c_string("G",e[1])
      elif 'D' in e:
        s=colors.c_string("R",e[1])
      acc+=n
      stdout.write("[Worst-%.2d] %.4f%% %.4f%% - %s\n" %(len(events)-1-i+1, float(n)/ref_count*100,float(acc)/ref_count*100, s.encode("utf-8")))

def main():
  cmd_parser = OptionParser(usage="usage: %prog [options] recognized_file reference_file")
  cmd_parser.add_option('-v',
      action="store_true",dest="v",
      help='Verbose power on!')
  cmd_parser.add_option('-V', '--verbose',
     action="store", type="int", dest="V", default=0, help='Verbose level')
  cmd_parser.add_option('-n', '--worst-events',
     action="store", type="int", dest="n", default=0, help='Words words to print')
  cmd_parser.add_option('-e', '--equal-func',
     action="store", type="string", dest="equal_func", default="standard", help='String compare function=[ '
     'standard , dummy, lower ]')
  cmd_parser.add_option('--cer',
     action="store_true", dest="cer", help='Calculate Character Error Rate')
  cmd_parser.add_option('-f', '--excp-file',
     action="store", type="string", dest="excp_file",  help='File containing the characters to delete')
  cmd_parser.add_option('-c', '--colors',
      action="store_true",dest="color",
      help='Color the output')
  cmd_parser.add_option('-O', '--vocab',
     action="store", type="string", dest="vocab", default=None, help='Vocabulary to count OOVs')
  cmd_parser.add_option('-K','--number-keys',
     action="store_true", dest="key_pressed", help='Calcultate the number of keys need to correct erroneous words')
  cmd_parser.add_option('-i', '--ignore-blank',
     action="store_true", dest="ignore_blank", help='Ignore blank lines in reference')

  cmd_parser.parse_args(argv)
  (opts, args)= cmd_parser.parse_args()


  if len(args) != 2:
    cmd_parser.print_help()
    exit(1)

  rec_file = codecs.open(args[0],"r","utf-8")
  ref_file = codecs.open(args[1],"r","utf-8")
  rec_file_reader = FileReader(rec_file)
  ref_file_reader = FileReader(ref_file)

  calculate_statistics(rec_file,ref_file,opts)

if __name__ == "__main__":
  main()

