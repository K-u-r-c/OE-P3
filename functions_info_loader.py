#!/usr/bin/env python3

import os, json, codecs, warnings
from packaging.version import Version

__author__      = 'Luca Baronti'
__maintainer__  = 'Luca Baronti'
__license__     = 'GPLv3'
__version__     = '1.1.3'

# this is the version of the json schema
CURRENT_VERSION = Version("0.1") 

class Optimum(object):
	def __init__(self, dvalues, optimum_type, score=None): # dvalues is something like ['+',[0.0, 0.0]]
		self.type = self._optima2optimum_type(optimum_type)
		if type(dvalues) is not list:
			raise ValueError(f"In creating local optimum expected a list for dvalues, found {dvalues}")
		if len(dvalues) != 2:
			raise ValueError(f"In creating local optimum expected a list of size 2 for dvalues, found {dvalues}")
		if type(dvalues[0]) is not str or type(dvalues[1]) is not list:
			raise ValueError(f"In creating local optimum expected a list [str, list] for dvalues, found {type(dvalues[0]),type(dvalues[1])}")
		self.position = dvalues[1]
		self.region_type = self._char2type(dvalues[0])
		self.score = score

	def _optima2optimum_type(self, optimum_type):
		if optimum_type not in ['minima', 'maxima', 'saddles']:
			raise ValueError(f"In creating local optimum expected a type to be either a minima, maxima or saddles, found {optimum_type}")
		if optimum_type=='minima':
			return 'Minimum'
		elif optimum_type=='maxima':
			return 'Maximum'
		else:
			return 'Saddle'
	def to_info_entry(self):
		return [self._type2char(self.region_type), self.position]
	def __str__(self):
		return str((self.score, self.position))
	def __len__(self):
		return len(self.position)
	def __lt__(self, optimum):
		return self.score<optimum.score
	def __le__(self, optimum):
		return self.score<=optimum.score
	def __gt__(self, optimum):
		return self.score>optimum.score
	def __ge__(self, optimum):
		return self.score>=optimum.score
	def __getitem__(self,index):
		return self.position[index]

	# Allowed chars are '+': (concave/convex) '_': (plateau) '~': (saddle/unknown)
	def _char2type(self, char):
		if char not in ['+', '_', '~']:
			raise ValueError(f"In creating local optimum expected an attraciton region type to be either +, _ or ~ found {char}")
		if char=='+':
			if self.type=='Minimum':
				return "Convex"
			elif self.type=='Maximum':
				return "Concave"
			else:
				raise ValueError(f"In creating local optimum a seddle point can not be part of a convex/concave region")
		elif char=='_':
			return "Plateau"
		else:
			if self.type=='Saddle':
				return 'Saddle'
			else:
				return 'Unknown'
	def _type2char(self, rtype):
		if rtype not in ["Convex", "Concave", "Plateau", 'Saddle', 'Unknown']:
			raise ValueError(f"In creating local optimum expected an attraciton region type to be either +, _ or ~ found {rtype}")
		if rtype in ["Convex", "Concave"]:
			return '+'
		elif rtype in ['Saddle', 'Unknown']:
			return '~'
		else:
			return '_'
		
class Reference(object):
	def __init__(self, raw_data):
		data=raw_data.strip()
		if data[0]!="@" or data[-1]!="}":
			raise ValueError("Problems in formatting the following reference: "+raw_data)
		self.paper_type = data[1:data.find("{")]
		content = data[data.find("{")+1:-1] # now content contains the whole reference, minus the e.g. "@book" part
		parts = content.split(',')
		# each part either contain a complete field or need to be merged with contiguous parts
		new_parts=[]
		for part in parts:
			if len(new_parts)==0 or new_parts[-1].count('{')==new_parts[-1].count('}'):
				new_parts+=[part]
			else:
				new_parts[-1]+=part
		parts=new_parts
		self.ref = parts[0]
		parts=parts[1:]
		self.fields={}
		for el in parts:
			tk = el.split('=')
			if len(tk)!=2:
				raise ValueError("Problems (tokens!=2) in formatting the following line of the reference: "+el)
			self.fields[tk[0].strip()]=self._format_field(tk[1])
		
	def to_latex(self):
		return '@'+self.paper_type+'{'+self.ref+',\n'+',\n'.join(['\t'+k+'='+'{'+self.fields[k]+'}' for k in self.fields.keys()])+'\n}'
	def _format_field(self, field):
		return field.replace('{','').replace('}','')
	def __str__(self):
		s=''
		if 'author' in self.fields:
			s+=self.fields['author']+', '
		if 'title' in self.fields:
			s+='"'+self.fields['title']+'" '
		if 'journal' in self.fields:
			s+=self.fields['journal']+', '
		if 'year' in self.fields:
			s+=self.fields['year']
		return s

class FunctionInfo(object):
	def __init__(self, fname):
		self.path = self.name2path(fname)
		self.config=None
		self.name = fname
		self.load()

	def name2path(self, fname):
		name = fname.lower().replace(',','').replace(' ','_')
		return os.path.join(os.path.dirname(os.path.abspath(__file__)),"functions_info", name+".json")

	def _parameters2str(self,parameters):
		return ','.join([k+'='+str(v) for k,v in parameters])

	def load(self):
		if not os.path.exists(self.path):
			raise ValueError(f"File {self.path} doesn't exist.")
		f=open(self.path,'r')
		self.config = json.load(f)
		f.close()
		if len(self.config.keys())!=1:
			raise ValueError(f"File {self.path} must contains a single function info, found {len(self.config.keys())}")
		if self.name.upper()!=list(self.config.keys())[0]:
			raise ValueError(f"In file {self.path} expected info for function '{self.name.upper()}', found {list(self.config.keys())[0]}")
		v = self.get_version()
		if not v is None and v<CURRENT_VERSION:
			warnings.warn(f"Function info file {self.path} has an outdated format (v{v} < v{CURRENT_VERSION}) therefore some info may not be present or accurate.")

	def _get_solutions(self,n_dimensions,optimum_type,parameters=[]):
		ret=[]		
		name=self.name.upper()
		if len(parameters)==0:
			if optimum_type not in self.config[name]:
				return []
			vals=self.config[name][optimum_type]
		else:
			pn=self._parameters2str(parameters)
			if pn not in self.config[name] or optimum_type not in self.config[name][pn]:
				return []
			vals=self.config[name][pn][optimum_type]
		if '*' in vals:
			ret+=[[vals['*'][0][0],[vals['*'][0][1][0]]*n_dimensions]] # this is just a workaround to make the call consistent
		if str(n_dimensions) in vals:
			ret+=vals[str(n_dimensions)]
		return [Optimum(r, optimum_type) for r in ret]
			
	# returns all the known minima
	def get_minima(self,n_dimensions,parameters=[]):
		return self._get_solutions(n_dimensions,"minima",parameters)

	# returns all the known maxima
	def get_maxima(self,n_dimensions,parameters=[]):
		return self._get_solutions(n_dimensions,"maxima",parameters)

	# returns all the known saddle points
	def get_saddles(self,n_dimensions,parameters=[]):
		return self._get_solutions(n_dimensions,"saddles",parameters)

	def _get_known_optima_number(self, optimum_type, parameters=[]):
		ret={}
		name=self.name.upper()
		if len(parameters)==0:
			if optimum_type not in self.config[name]:
				return {}
			vals=self.config[name][optimum_type]
		else:
			pn=self._parameters2str(parameters)
			if pn not in self.config[name] or optimum_type not in self.config[name][pn]:
				return {}
			vals=self.config[name][pn][optimum_type]
		for d in vals:
			if d=='*':
				ret[d]=len(vals[d])
			else:
				ret[int(d)]=len(vals[d])
		return ret

	def get_number_minima(self, parameters=[]):
		return self._get_known_optima_number("minima", parameters)

	def get_number_maxima(self, parameters=[]):
		return self._get_known_optima_number("maxima", parameters)
	
	def get_number_saddles(self, parameters=[]):
		return self._get_known_optima_number("saddles", parameters)
	
	def get_suggested_bounds(self, parameters=[]):
		name=self.name.upper()
		if len(parameters)==0:
			vals=self.config[name]["suggested bounds"]
		else:
			pn=self._parameters2str(parameters)
			vals=self.config[name][pn]["suggested bounds"]
		return (vals["lower"], vals["upper"])

	def get_reference(self):
		name=self.name.upper()
		if name not in self.config or "reference" not in self.config[name] or self.config[name]["reference"]=='':
			return None
		return Reference(self.config[name]["reference"])
	
	def get_summary(self):
		name=self.name.upper()
		if name not in self.config or "summary" not in self.config[name] or self.config[name]["summary"]=='':
			return None
		return self.config[name]["summary"]

	def get_dev_comment(self):
		name=self.name.upper()
		if name not in self.config or "dev_commment" not in self.config[name] or self.config[name]["dev_commment"]=='':
			return None
		return self.config[name]["dev_commment"]

	def get_version(self):
		name=self.name.upper()
		if name not in self.config or "version" not in self.config[name] or self.config[name]["version"]=='':
			return None
		return Version(self.config[name]["version"])

	def get_definition(self):
		name=self.name.upper()
		if name not in self.config or "definition" not in self.config[name] or self.config[name]["definition"]=='':
			return None
		return codecs.decode(self.config[name]["definition"],'unicode_escape') # this is due to the conversion of escape chars \\ to \


class FunctionInfoWriter(FunctionInfo):
	# solution must be instance of Optimum
	def add_solution(self, solution, parameters=[], any_dimension=False):
		name=self.name.upper()
		if any_dimension:
			n_dimensions='*'
		else:
			n_dimensions=str(len(solution.position))
		if solution.type=='Minimum':
			optimum_type = 'minima'
		elif solution.type=='Maximum':
			optimum_type = 'maxima'
		else:
			optimum_type = 'saddles'
		src = self.config[name]
		if len(parameters)>0:
			pn=self._parameters2str(parameters)
			if pn not in src:
				src[pn]={
					'minima': {},
					'maxima': {},
					'saddles': {}
				}
				src=src[pn]
		src = src[optimum_type]
		if n_dimensions not in src:
			src[n_dimensions]=[]
		src[n_dimensions]+=[solution.to_info_entry()]

	def set_reference(self, reference):
		name=self.name.upper()
		self.config[name]['reference']=reference.to_latex()
	
	def set_summary(self, summary):
		name=self.name.upper()
		self.config[name]['summary']=summary

	def set_dev_comment(self, comment):
		name=self.name.upper()
		self.config[name]['dev_comment']=comment

	def set_definition(self, definition):
		name=self.name.upper()
		self.config[name]['definition']=definition
		
	def save(self):
		f = open(self.path,'w')
		json.dump(self.config, f, sort_keys=True, indent=1)
		f.close()