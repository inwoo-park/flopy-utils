#!/usr/bin/env python
'''
Explain
 oraganizer is originated from "organizer.m" in matlab function.

Created by
 Inwoo Park (inwoo0415@snu.ac.kr or inwoopark0415@gmail.com)
'''

class organizer: # {{{
    '''
    Explain
     default variables for organizer class.

    Usage
     steps = [1]  \n
     org = organizer(steps=steps) \n
     \n
     if perform(org,'Step1'): \n
         # something \n
     if perform(org,'Step2'): \n
         # something \n

    See also.
     perform
    '''
    id         = []
    prefix     = []
    repository = []
    steps      = []
    string     = []
    # }}}
    def __init__(self,steps=[],repository=[],prefix=[]): # {{{
        self.string = []
        self.steps = steps;
        self.repository = repository
        # }}}
    def __repr__(self):# {{{
        output = []
        for i in range(len(self.string)):
            output.append('steps#%d : %s\n'%(self.id[i],self.string[i]))
        return "".join(output);
        # }}}

def perform(org,string): # {{{
    '''
    Explain
     Without ordering each step, we can select specific step with this function.

    Usage
        >> steps = [1]

        >> org = organizer('steps',steps)

        >> if perform(org,'step#1_~~'):

           # write someting below.

        >> if perform(org,'step#2_~~'):

           # write someting below.
    '''
    if not isinstance(org, organizer):
        help(perform)
        raise Exception('input type is not "organizer". See usage')
    #print(string)
    # count numer of previous step
    cn = len(org.string)+1

    # append string and id.
    org.string.append(string)
    org.id.append(cn)

    # is it current step?
    #print(steps)
    if cn in org.steps:
        print('  Step# %d : %s'%(cn,string))
        return True
    else:
        return False
    # }}}

