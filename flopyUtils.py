#!/usr/bin/env python
'''
Explain
 "flopyUtils.py" supports "Flopy" as user defined functions.

'''
# for system command
import glob, os, sys, shutil, importlib

import flopy
import pandas
import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import traceback
import geopandas
import shapely
import netCDF4

# for changing ESPG system of modflow.
import pyproj # currently use this function
try:          # try to import osgeo for futhers
    import osgeo
    from osgeo import ogr, osr
except:
    raise Warning("cannot load osgeo modules.")

# download module
import wget # for downloading usgs package.

# geopandas
from shapely.geometry import Point, LineString
import inspect

# use TKAgg at linux machine # {{{
if sys.platform == 'linux':
    matplotlib.use('TKAgg')
# }}}

# utils
def mfPrint(string,debug=False):# {{{
    '''

    Usage:
        >>> mfPrint('write something',debug=debug)

    Variables\n
        string : show variables \n
        debug  : show or not.\n
    '''
    if debug:
        print(string)
# }}}
def print_(string,debug=False):# {{{
    '''

    Usage:
        >>> mfPrint('write something',debug=debug)

    Variables\n
        string : show variables \n
        debug  : show or not.\n
    '''
    if debug:
        mfPrint(string,debug=1)
# }}}
def mf96Print(string,debug=False): # {{{
    '''
    Explain
     Print debuging. This function same as "mfPrint".

    Usage
        >>> mf96Print("write something",debug=debug)

    See also.
     mfPrint,
    '''
    mfPrint(string,debug=debug)
# }}}

# option util
def varargout(*args): # {{{
    '''
    Explain
     this function is enable to variable outputs.

    Usage
        def func():\n
             a = 1\n
             b = 2\n
             return varargout(a,b)\n
        \n
        a = func() \n
        a,b = func()\n

    Reference
     https://stackoverflow.com/questions/14147675/nargout-in-python
    '''
    callInfo = traceback.extract_stack()
    callLine = str(callInfo[-3].line)
    split_equal = callLine.split('=')
    split_comma = split_equal[0].split(',')
    num = len(split_comma)
    return args[0:num] if num > 1 else args[0]
# }}}
def dictoptions(options): #{{{
    if not isinstance(options,dict):
        raise Exception('ERROR: option is not "dict" type.')
    return options
# }}}
def getfieldvalue(options,string,default_value): # {{{
    '''
    Explain

    Usage:
        options = dictoptions(kwargs)
    '''
    # check option type.
    if not isinstance(options,dict):
        raise Exception('ERROR: option is not "dict" type.')

    if string in options.keys():
        return options[string]
    else:
        options.update({string:default_value})
        return default_value

# }}}
def addfieldvalue(options,string,default_value): # {{{
    '''
    Explain

    Usage
     options = dictoptions(kwargs)
     fontsize = addfieldvalue(options,'fontsize',12)
    '''
    # check option type.
    if not isinstance(options,dict):
        raise Exception('ERROR: option is not "dict" type.')

    if string in options.keys():
        options[string]=default_value
    else:
        options.update({string:default_value}) 
    return options
# }}}

# Update GMS model.
# def updateGMS {{{
def updateGMS(mfinput,model_ws='./',modelname='modflowtest',exe_name='mf2005',
        version='mf2005',debug=False):
    '''
     Explain
      Update modflow from GMS.

     Usage
      mfGMS = flopy.modflow.Modflow.load('./GMSname.mfn')
      mf = updateGMS(mfGMS, model_ws = './Transient/',modelname='modflowtest')
    '''
    # update solver version.
    if exe_name is not version:
        version = exe_name
    
    # temporary modflow 
    mf = flopy.modflow.Modflow(modelname=modelname,
            model_ws=model_ws, exe_name=exe_name,version=version)

    # update dis
    mf.dis = updateDis(mfinput)
    
    # update bas
    #mf.bas6 = mfinput.bas6
    flopy.modflow.ModflowBas(mf, ibound = mfinput.bas6.ibound, strt = mfinput.bas6.strt)

    # update lpf 
    if debug:
        print('show lpf hk shape')
        print(mfinput.lpf.hk.shape)
        print(mfinput.lpf.vka.shape)
        print(mfinput.lpf.ss.shape)
        print(mfinput.lpf.sy.shape)
        print(mfinput.lpf.laytyp.shape)
        print(mfinput.lpf.chani.array)
    flopy.modflow.ModflowLpf(mf,hk = mfinput.lpf.hk, vka = mfinput.lpf.vka, sy = mfinput.lpf.sy,
            ss = mfinput.lpf.ss, laytyp = mfinput.lpf.laytyp, layavg= mfinput.lpf.layavg,
            hani = mfinput.lpf.hani,
            chani = mfinput.lpf.chani, layvka = mfinput.lpf.layvka, laywet= mfinput.lpf.laywet,
            )

    # update riv
    #mf.riv = mfinput.riv
    stress_period_data = {}
    for i in range(len(mfinput.riv.stress_period_data.data)):
        data = mfinput.riv.stress_period_data.data[i][['k','i','j','stage','cond','rbot']]
        stress_period_data.update({i:pandas.DataFrame(data).to_numpy()})

    flopy.modflow.ModflowRiv(mf, stress_period_data = stress_period_data)

    # update wel package
    stress_period_data = {}
    for i in range(len(mfinput.wel.stress_period_data.data)):
        data = pandas.DataFrame(mfinput.wel.stress_period_data.data[i][['k','i','j','flux']])
        stress_period_data.update({i:pandas.DataFrame(data).to_numpy()})
    flopy.modflow.ModflowWel(mf, stress_period_data = stress_period_data)
    
    # update pcg package
    #mf.pcg = flopy.modflow.ModflowPcg(mf) # old
    flopy.modflow.ModflowPcg(mf)

    # update oc package
    #mf.oc = mfinput.oc # old
    flopy.modflow.ModflowOc(mf,stress_period_data=mfinput.oc.stress_period_data)

    # update rch package
    mf.rch = mfinput.rch

    # output
    return mf
# }}}
# def updateDis {{{
def updateDis(mf,nper=[],perlen=[],nstp=[],steady=[],
        nlay=[],ncol=[],nrow=[],delr=[],delc=[],laycbd=[],top=[],botm=[],
        proj4_str=[],xul=[],yul=[],debug=False):
    '''
    Explain
     change specific variables of DIS package.

    Usage
     dis = updateDis(mf,nper=nper,perlen)
    '''
    f_name = inspect.currentframe().f_code.co_name

    # time variables{{{
    if not nper:
        nper = mf.dis.nper
    if not perlen:
        perlen = mf.dis.perlen
    if not nstp:
        nstp = mf.dis.nstp
    if not steady:
        steady = mf.dis.steady
    # }}}

    # geometry variables {{{
    if not nlay:
        nlay = mf.dis.nlay
    if not ncol:
        ncol = mf.dis.ncol
    if not nrow:
        nrow = mf.dis.nrow
    if not delr:
        delr = mf.dis.delr
    if not delc:
        delc = mf.dis.delc
    if not laycbd:
        laycbd = mf.dis.laycbd.array
    if not top:
        top = mf.dis.top.array
    if not botm:
        botm = mf.dis.botm.array
    # }}}

    # projection references {{{
    if not proj4_str:
        proj4_str= mf._proj4_str
    if proj4_str is not None and "EPSG" in proj4_str:
        #print('convert from EPSG = %s'%(proj4_str))
        crs = pyproj.CRS.from_string(proj4_str)
        proj4_str = crs.name

    # left-upper corner grid point.
    xlen, ylen = flopyGetXylength(mf) # get xy length of grid.
    if not xul:
        if flopy.__version__ >= '3.3.4':
            xul = mf.modelgrid.xoffset
        else:
            xul = mf._xul
    if not yul:
        if flopy.__version__ >= '3.3.4':
            yul  = mf.modelgrid.yoffset + ylen
        else:
            yul= mf._yul
    print_('   {}: xul = {}'.format(f_name,xul),debug=debug)
    print_('   {}: yul = '.format(f_name,yul),debug=debug)
    # }}}

    # update dis file
    mf.dis = flopy.modflow.ModflowDis(mf,nper=nper, perlen=perlen, nstp=nstp, steady=steady,
            nlay = nlay, ncol = ncol, nrow=nrow,
            delr = delr, delc=delc,
            laycbd = laycbd,
            top = top,
            botm = botm,
            proj4_str=proj4_str, xul=xul, yul=yul)

    if flopy.__version__ == '3.3.4':
        mf._xul = xul
        mf._yul = yul

    return mf.dis
    # }}}
# def updateBas {{{
def updateBas(mf,ibound=[], strt=[]):

    if not ibound:
        ibound = mf.bas6.ibound.array
    if isinstance(strt,numpy.ndarray):
        mfPrint('updateBas: keep use numpy array.',debug=mf.verbose)
    elif not strt :
        strt = mf.bas6.strt.array

    mf.bas = flopy.modflow.ModflowBas(mf, ibound = ibound, strt = strt)
    return
# }}}
def updateLpf(mf,hk=[],vka=[],ss=[],sy=[],hani=[],vkcb=[],wetdry=[]):# {{{
    if not np.any(hk):
        hk = mf.lpf.hk.array

    if not vka:
        vka = mf.lpf.vka.array
    if not np.any(ss):
        ss = mf.lpf.ss.array
    if not sy:
        sy = mf.lpf.sy.array

    mf.lpf = flopy.modflow.ModflowLpf(mf,hk=hk,vka=vka,ss=ss,sy=sy)
    return mf.lpf
# }}}
def updateGhb(mf,stress_period_data=[],ipakcb=[]): # {{{
    if not np.any(ipakcb):
        ipakcb = mf.ghb.ipakcb
    if not np.any(stress_period_data):
        print('none')
    # }}}
def updateBtn(mt,sconc=[],prsity=[],thkmin=[],munit=[],icbund=[],nprs=[]): # {{{
    if not np.any(sconc):
        sconc = mt.btn.sconc
    if not np.any(prsity):
        prsity = mt.btn.prsity

    btn = flopy.mt3d.Mt3dBtn(mt,sconc=sconc,prsity=prsity,thkmin=thkmin,
            munit=munit,icbund=icbund,nprs=nprs)
    # }}}
def updateEPSG(mf,epsg_src,epsg_tgt,debug=0): # {{{
    '''
    Explain
     change EPSG system of modflow.

    Usage
     epsg_src = '5174'
     epsg_tgt = '5186'
     mf = updateEPSG(mf,epsg_src,epsg_tgt)
    '''
    f_name = inspect.currentframe().f_code.co_name

    # get x,y coordinates of modflow.
    mx,my = flopyGetXY(mf,center=0,debug=debug)

    if 0: #old version
        # change EPSG coordinates of model.
        src = gdal.osr.SpatialReference()
        tgt = gdal.osr.SpatialReference()
        src.ImportFromEPSG(epsg_src)
        tgt.ImportFromEPSG(epsg_tgt)

        transform = gdal.osr.CoordinateTransformation(src,tgt)
        print_('   %s: x0,y0 = (%f,%f)'%(f_name,mx[0],my[0]),debug=debug)
        yul, xul = transform.TransformPoint(my[0],mx[0])
    else:
        # check pyproj version
        if pyproj.__version__ >= '3.0.0':
            inProj = pyproj.Proj(epsg_src)
            outProj = pyproj.Proj(epsg_tgt)
            yul, xul = pyproj.transform(inProj,outProj,my[0],mx[0])
        else:
            print('current pyproj {} is not supported'.format(pyproj.__version__))

    # update xul, yul
    if flopy.__version__ == '3.3.4':
        print_('   {}: update EPSG = {},{}'.format(f_name,xul,yul),debug=debug)
        xlen, ylen = flopyGetXylength(mf)
        mf.modelgrid._xoff = xul
        mf.modelgrid._yoff = yul-ylen
    else:
        updateDis(mf,xul=xul, yul=yul)

    # output
    return varargout(mf)
# }}}
def ChangeOutputName(mf,prefix=[]): # {{{
    '''
    Explain:
    --------
     When load modflow through flopy (mf = flopy.modflow.Modflow.load('test/test.nam')), the output model is not changed. Therefore, the output file names are changed mannually.

    Usage:
    -----
        mf = flopy.modflow.Modflow.load('test/test.nam')
        mf.name = 'test01'
        mf.model_ws = './test01/'
        mf = ChangeOuputName(mf,'test01')
        #or
        mf = ChangeOutputName(mf) # update output name from 'mf.name'
    '''
    # check input variables.
    if not prefix:
        prefix = mf.name

    for i, fname in enumerate(mf.output_fnames):
        surfix = fname.split('.')[1]
        mf.output_fnames[i] = prefix + '.' + surfix

    return mf
# }}}

# Grid and index functions.
def flopyGetXY(mf,center=1,debug=False): # {{{
    '''
    Explain
     Get x,y grid points from modflow model.
     :math:`x \in R^n`
     :math:`y \in R^m`

    Usage
    x,y = flopyGetXY(mf)

    # get centered x,y poistion
    x,y   = flopyGetXY(mf)
    x,y,z = flopyGetXY(mf)

    '''
    # define function name.
    f_name = inspect.currentframe().f_code.co_name

    if not isinstance(mf,flopy.modflow.mf.Modflow):
        raise Exception('Error: input file is not type(flopy.modflow.Modflow.mf')
    nlay = mf.dis.nlay
    ncol = mf.dis.ncol
    nrow = mf.dis.nrow

    # [nlay, nrow, ncol]
    dx = mf.dis.delr.array
    dy = mf.dis.delc.array
    x = numpy.zeros(dx.shape[0]+1)
    y = numpy.zeros(dy.shape[0]+1)
    x[1:] = dx.cumsum() # cumulative sum for calculating x based on dx.
    y[1:] = dy.cumsum() # cumulative sum for calculating y based on dy.

    # get coorner coordinates
    print_('   %s: get global coordinate'%(f_name),debug=debug)
    if flopy.__version__ <= '3.3.3':
        print_('   {}: flopy version = {}'.format(f_name,flopy.__version__),debug=debug)
        xul = mf.dis._sr.xul # upper left corner grid
        yul = mf.dis._sr.yul # upper left corder grid
    elif flopy.__version__ == "3.3.4":
        print_('   {}: flopy version = {}'.format(f_name,flopy.__version__),debug=debug)
        xul = mf.modelgrid.xoffset# upper left corner grid
        yul = mf.modelgrid.yoffset+np.sum(dy)# upper left corder grid
    else:
        print_('   current version(flopy {}) is not available'.format(flopy.__version__),
                debug=1)
    
    print_('   {}: xul = {}'.format(f_name,xul),debug=debug)
    print_('   {}: yul = {}'.format(f_name,yul),debug=debug)


    # calculate centered coorinates, because modflow use block centered method.
    if center:
        x = (x[0:-1]+x[1:])/2
        y = (y[0:-1]+y[1:])/2

    # calibarte global coordinate with xul, yul.
    print_('   {}: len x = {}'.format(f_name,np.shape(x)),debug=debug)
    print_('   {}: len y = {}'.format(f_name,np.shape(y)),debug=debug)
    if xul:
        print_('   xul = %f'%(xul),debug=debug)
        x = x + xul
    if yul:
        print_('   yul = %f'%(yul),debug=debug)
        y = -y + yul

    # get z elevation from botm and top elevation.
    z    = np.zeros((nlay,nrow,ncol),dtype=float)
    botm = mf.dis.botm.array
    top  = mf.dis.top.array

    # calculate z elevation based on block centered method.
    print_(['top  array shape = ',np.shape(top)],debug=debug)
    print_(['botm array shape = ',np.shape(botm[0,:,:])],debug=debug)

    # top elevation
    z[0,:,:] =  (botm[0,:,:]+top)/2
    # bottom elevation
    for i in range(1,nlay):
        z[i,:,:] = np.mean(botm[i:i+1,:,:],axis=0)

    return varargout(x,y,z)
    # }}}
def flopyGetXyz(mf,center=1,debug=False): # {{{
    r'''
    Explain
     Get x y z grid coordinates from "dis" package.

     :math:`x \in R^{n \times m}`
     :math:`y \in R^{n \times m}`

    Usage
     x,y,z = flopyGetXyz(mf)
    '''
    x,y,z = flopyGetXY(mf,center=center,debug=debug)
    return x,y,z
# }}}
def flopyGetXyzGrid(mf,center=0,debug=False): # {{{
    '''
    Explain
     Get x,y meshgrid points from modflow model.

    Usage
     # get un-centered x,y grid position
     xg,yg = flopyGetXyzGrid(mf)

     # get centered x,y grid poistion
     xg,yg,zg = flopyGetXyzGrid(mf,center=1)

    '''
    if not isinstance(mf,flopy.modflow.mf.Modflow):
        raise Exception('Error: input file is not type(flopy.modflow.Modflow.mf')

    if 0:
        # old version for get x,y coordinates {{{
        nlay = mf.dis.nlay
        ncol = mf.dis.ncol
        nrow = mf.dis.nrow

        # get coorner coordinates
        xul = mf.dis._sr.xul # upper left corner grid
        yul = mf.dis._sr.yul # upper left corder grid

        # [nlay, nrow, ncol]
        dx = mf.dis.delr.array
        dy = mf.dis.delc.array
        x = numpy.zeros(dx.shape[0]+1)
        y = numpy.zeros(dy.shape[0]+1)
        x[1:] = dx.cumsum()
        y[1:] = dy.cumsum()

        # calculate centered coorinates, because modflow use block centered method.
        if center:
            x = (x[0:-1]+x[1:])/2
            y = (y[0:-1]+y[1:])/2

        # calibarte global coordinate with xul, yul.
        if debug:
            print('len x = ',x.shape)
            print('len y = ',y.shape)

        if xul:
            x = x + xul
        if yul:
            y = -y + yul

        # get z elevation from geometry
        bot = mf.dis.botm.array
        top = mf.dis.top.array
        top = top.reshape((1,nrow,ncol))
        mfPrint(['   nlay = ',mf.dis.nlay] ,debug=debug)
        mfPrint(['   bot = ',np.shape(bot)],debug=debug)
        mfPrint(['   top = ',np.shape(top)],debug=debug)
        z = np.concatenate((top,bot),axis=0)
        # }}}
    else:
        x,y,z = flopyGetXyz(mf,center=center,debug=debug)

    # generate mesh grid.
    x,y = np.meshgrid(x,y)

    return varargout(x,y,z)
# }}}
def flopyGetXyGrid(mf,center=0,debug=False): # {{{
    '''
    Explain
     get 2d xy grid array.
    '''
    xg,yg = flopyGetXyzGrid(mf,center=center,debug=debug)
    return varargout(xg,yg)
# }}}
def flopyGetXylength(mf,debug=False):# {{{
    '''
    Explain
     get length of modflow domain.

    Usage
     xlen, ylen = flopyGetXylength(mf)
    '''
    # define function name.
    f_name = inspect.currentframe().f_code.co_name

    if not isinstance(mf,flopy.modflow.mf.Modflow):
        raise Exception('Error: input file is not type(flopy.modflow.Modflow.mf')

    # [nlay, nrow, ncol]
    xlen = np.sum(mf.dis.delr.array)
    ylen = np.sum(mf.dis.delc.array)

    return xlen, ylen
    # }}}

def flopyIndexToXy(mf,cols,rows,lays,debug=False):# {{{
    '''
    Explain
     Get x y coordinates from (nlay, nrow, ncol) array.
                               z      y    x

    Usage
     wel = mf.wel.stress_period_data[0]
     x,y = flopyIndexToXy(mf,wel[2],wel[1],wel[0])

     # others
     x,y,z = flopyIndexToXy(mf,cols,rows,lays)

    Options
     debug - show process of current function.
    '''

    # get x,y,z coordinates in mf
    xi,yi,zi = flopyGetXY(mf,center=1)

    # force dtype as int
    if not isinstance(cols,list):
        cols = cols.astype(int)
    if not isinstance(rows,list):
        rows = rows.astype(int)
    if not isinstance(lays,list):
        lays = lays.astype(int)

    mfPrint('   cols = {0}'.format(cols),debug=debug)
    mfPrint('   rows = {0}'.format(rows),debug=debug)

    if not np.shape(lays):
        mfPrint('   Force lays',debug=debug)
        mfPrint('   check shape of cols = {0}'.format(np.shape(cols)),debug=debug)
        lays = np.zeros(np.shape(cols))

    # check length of array
    if np.shape(lays) != np.shape(rows) or np.shape(lays) != np.shape(cols) or np.shape(rows) != np.shape(cols):
    #if len(lays) != len(rows) or len(lays) != len(cols) or len(rows) != len(cols):
        raise Exception('ERROR: check length of x,y,z.')

    # initialize x,y points
    x = np.zeros(np.shape(cols))
    y = np.zeros(np.shape(rows))
    z = np.zeros(np.shape(lays))
    if not np.shape(cols):
        x = xi[int(cols)]
        y = yi[int(cols)]
        z = zi[int(lays), int(rows), int(cols)]
    else:
        for i in range(np.shape(cols)[0]):
            x[i] = xi[int(cols[i])]
            y[i] = yi[int(rows[i])]
            z[i] = zi[int(lays[i]), int(rows[i]), int(cols[i])]

    return varargout(x,y,z)
# }}}
def flopyIndexToGrid(mf,cols,rows,lays,values,debug=False):# {{{
    '''
    Explain
     generate grid data from point values.

    Usage
     grid_output = flopyIndexToGrid(mf,cols,rows,lays,value)
)
    '''

    # get domain
    nrow = mf.dis.nrow.value
    ncol = mf.dis.ncol.value
    nlay = mf.dis.nlay.value

    # initialize output grid variables.
    output = np.zeros((nlay,ncol,nrow),dtype=float)

    for i in range(np.shape(values)[0]):
        output[lays[i],rows[i],cols[i]] = values[i]

    return varargout(output)
# }}}
def flopyXyToIndex(mf,x,y,debug=False):# {{{
    '''
    Explain
     
    '''
    cols, rows, lays = flopyXyzToIndex(mf,x,y,[])

    # outputs
    return varargout(cols, rows)
# }}}
def flopyXyzToIndex(mf,x,y,z,debug=False):# {{{
    '''
    Explain
     Get index for modflow from x,y,z coordinates.

     nlay: 0-for top, end of nlay-bottom.

    Usage
     cols, rows, lays = flopyXyToIndex(mf,wx,wy,wz)
    '''

    # check input types
    if isinstance(x,float) or isinstance(x,int) or isinstance(x,numpy.int64) or isinstance(x,numpy.float64):
        x = [x]
    if isinstance(y,float) or isinstance(y,int) or isinstance(y,numpy.int64) or isinstance(y,numpy.float64):
        y = [y]
    if not np.any(z): 
        z = np.zeros(np.shape(x),dtype=float)
    if isinstance(z,float) or isinstance(z,int):
        z = [z]

    # check length of each variables
    mfPrint(type(x),debug=debug)
    if isinstance(x,pandas.core.series.Series):
        x =x.to_numpy()
    if isinstance(y,pandas.core.series.Series):
        y =y.to_numpy()
    if isinstance(z,pandas.core.series.Series):
        z =z.to_numpy()

    mfPrint('   len(x) = %d'%(len(x)),debug=debug)
    mfPrint('   len(y) = %d'%(len(y)),debug=debug)
    mfPrint('   len(z) = %d'%(len(z)),debug=debug)
    if len(x) != len(y) or len(x) != len(z) or len(y) != len(z):
        raise Exception('ERROR: Check length of x,y,z coordinates.')

    # get x,y,z grid information
    xg,yg,zg = flopyGetXyzGrid(mf,center=1)

    mfPrint('zg shape = {}'.format(np.shape(zg)),debug=debug)

    mfPrint('   find poisition',debug=debug)
    cols = np.zeros((len(x),1),dtype=int) # z dir
    rows = np.zeros((len(y),1),dtype=int) # y dir
    lays = np.zeros((len(z),1),dtype=int) # z dir
    for i in range(len(x)):
        dist = np.square((xg-x[i])**2 + (yg-y[i])**2)
        cols[i] = dist.argmin(axis=1).min()
        rows[i] = dist.argmin(axis=0).min()
        dist_z = np.square((zg[:,rows[i],cols[i]]-z[i])**2)
        #print(dist_z)
        lays[i] = dist_z.argmin()
        #print(zg[:,rows[i],cols[i]], z[i])

        # show rows and cols
        print_('   rows = %4d cols = %4d'%(rows[i],cols[i]),debug=debug)

    # outputs
    return varargout(cols, rows, lays)
# }}}

# export and import netcdf data
def flopysave(mf,filename=[],headobj=[],concobj=[]): # {{{
    '''
    Explain
     export modflow with flopy as "netcdf" format('.nc')

    Usage
     flopysave(mf,'./output.nc')
    '''

    # check filename
    if not filename:
        filename = mf.model_ws + '/' + mf.name + '.hds'
    
    # get model grid information
    nlay = mf.dis.nlay
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol

    # get xyz grid
    [x,y,z] = flopyGetXY(mf)

    # check head data
    headname = mf.model_ws + '/' + mf.name +'.hds'
    head     = []
    times    = []
    if headname:
        headobj = flopy.utils.binaryfile.HeadFile(headname)
        times = headobj.get_times()
        for i, time in enumerate(times):
            head.append(headobj.get_data(totim=times[-1]))

    concname = mf.model_ws + '/MT3D001.UCN'
    conc    = []
    if concname:
        concobj = flopy.utils.binaryfile.UcnFile(concname)
        times = concobj.get_times()
        for i, time in enumerate(times):
            conc.append(concobj.get_data(totim=times[-1]))
    
    print('check head shape')
    print(np.shape(head))
    print('check head shape')
    print(np.shape(z))
    print('check timeshape')
    print(np.shape(times))

    ncfile = netCDF4.Dataset(filename,mode='w',format='NETCDF4')

    # set default dimensions.
    ncfile.createDimension('nlay',nlay)
    ncfile.createDimension('nrow',nrow)
    ncfile.createDimension('ncol',ncol)
    if times:
        ncfile.createDimension('ntime',len(times))

    # export grid imformation
    ncx = ncfile.createVariable('x',np.float,('ncol',))
    ncy = ncfile.createVariable('y',np.float,('nrow',))
    ncz = ncfile.createVariable('z',np.float,('nlay','nrow','ncol',))
    nctimes = ncfile.createVariable('times',np.float,('ntime',))
    ncx.units = 'm'
    ncy.units = 'm'
    ncz.units = 'm'
    nctimes.units = 'day'
    ncx[:] = x
    ncy[:] = y
    ncz[:] = z
    nctimes[:] = times

    # write results.
    nchead = ncfile.createVariable('head',np.float,('ntime','nlay','nrow','ncol',))
    ncconc = ncfile.createVariable('conc',np.float,('ntime','nlay','nrow','ncol',))
    nchead[:]  = head
    ncconc[:]  = conc

    # close nc file
    ncfile.close()
# }}}

# Plot results from Flopy.
def plotFlopy3d(mf,data,**kwargs): #{{{
    '''
    Explain
     This function helps to plot data from flopy.

    Usage:
        # plot 2d on top
        plotFlopy3d(mf,h ead,layer=0,title='layer 0(top)')

        # plot 2d on bottom
        plotFlopy3d(mf,head,layer=1,title='layer 1(bottom)')

        # add figure on specific axes.
        plotFlopy3d(mf,head,layer=0,ax=ax1)
    '''
    f_name = inspect.currentframe().f_code.co_name

    options  = dictoptions(kwargs)
    debug    = getfieldvalue(options,'debug',False)
    layers   = getfieldvalue(options,'layer',0) # layer=0 for top layer
    ax       = getfieldvalue(options,'ax',[])
    print_('   {}: check axes'.format(f_name),debug=debug)
    print_('   {}: {}'.format(f_name,ax,debug=debug))
    print_('   {}: {}'.format(f_name,type(ax)),debug=debug)
    fontsize = getfieldvalue(options,'fontsize',8)
    xlabel   = getfieldvalue(options,'xlabel','x (m)')
    ylabel   = getfieldvalue(options,'ylabel','y (m)')
    title    = getfieldvalue(options,'title',[])

    # initial geometry information from "modflow"
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    nlay = mf.dis.nlay

    # determine 3d or 2d from layer
    if isinstance(layers,int):
        is3d = 0
    else:
        is3d = 1
    print_('   %s: is3d = %d'%(f_name,is3d),debug=debug)

    # determine figure and axes.
    if not ax:
        mfPrint('   define new figure.',debug=debug)
        fig = plt.figure()
        if is3d:
            ax = fig.add_subplot(1,1,1,projection='3d')
        else:
            ax = fig.add_subplot(1,1,1)
        addfieldvalue(options,'ax',ax)
    else:
        fig = plt.gcf()

    # set plot grid.
    x,y = flopyGetXY(mf,debug=debug)
    xg = (x[0:-1]+x[1:])/2
    yg = (y[0:-1]+y[1:])/2
    xg,yg = numpy.meshgrid(xg,yg)

    # set x,y limit.
    xlim = getfieldvalue(options,'xlim',[np.min(x), np.max(x)])
    ylim = getfieldvalue(options,'ylim',[np.min(y), np.max(y)])

    # get geometry information
    botm = mf.dis.botm.array

    mfPrint(['xg = ',xg.shape,'  yg = ',yg.shape],debug=debug)

    # processing data type
    if isinstance(data,np.ndarray):
        mfPrint(['   data type as numpy and ',data.dtype],debug=debug)
        if data.dtype == 'int32':
            mfPrint('   integer to float',debug=debug)
            data = data.astype('float')
        else:
            mfPrint('   not integer',debug=debug)

        # check data array just (nrow, ncol)
        # change (nrow, ncol) to (1,nrow,ncol)
        s = np.shape(data)
        if(nrow==s[0] and ncol==s[1]):
            data = np.reshape(data,(1,nrow,ncol))

    # masking: load ibound for maksing
    # 0 - inactive cell, -1 - constant head, 1 - simulation area.
    ibound = mf.bas6.ibound.array

    # processing find nan value in data.
    for i in range(np.shape(data.shape)[0]):
        data[i][data[i] <= -999.0+0.01] = numpy.nan
        data[i][ibound[i]==0] = np.nan


    # plot 3D plot
    if isinstance(layers,int): # plot 2D map
        mfPrint('   plot2d layer = %d'%(layers),debug=debug)
        data_lay = data[layers]

        # find min max value in data for colorbar ticks.
        caxis = getfieldvalue(options,'caxis',[np.nanmin(data_lay), np.nanmax(data_lay)])
        data_min = caxis[0]
        data_max = caxis[1]
        mfPrint('   cmax=%f,  cmin=%f'%(data_max, data_min),debug=debug)
        colors = plt.cm.jet((data_lay-data_min)/(data_max-data_min))

        # show 2d surface map.
        surf = ax.pcolor(x,y,data_lay,vmin=data_min,vmax=data_max,shading='auto',
                cmap='jet',edgecolor='none')
    else:
        ax = fig.add_subplot(1,1,1,projection='3d')
        # find min max value in data for colorbar ticks.
        caxis = getfieldvalue(options,'caxis',[np.nanmin(data), np.nanmax(data)])
        data_min = caxis[0]
        data_max = caxis[1]
        mfPrint('   cmax=%f,  cmin=%f'%(data_max, data_min),debug=debug)

        for i in layers:
            mfPrint('   plot3d layer = %d'%(i))
            data_lay = data[i,:,:]
            botm_lay = botm[i,:,:]
            botm_lay[ibound[i]==0] = numpy.nan
            colors = plt.cm.jet((data_lay-data_min)/(data_max-data_min))

            print_(['data  shape = ',numpy.shape(data_lay)],debug=debug)
            print_(['botm  shape = ',numpy.shape(botm_lay)],debug=debug)
            print_(['color shape = ',numpy.shape(colors)],debug=debug)
            print_(['jet   shape = ',numpy.shape(plt.cm.jet)],debug=debug)

            surf = ax.plot_surface(xg,yg,botm_lay,linewidth=0,
                    antialiased=False,cmap='jet')
            #surf = ax.plot_surface(xg,yg,botm_lay,facecolors=colors,linewidth=0,
            #        antialiased=False,cmap='jet')

    # set colorbar.
    cb = fig.colorbar(surf,ax = ax)

    if not isinstance(layers,int):
        mfPrint('   set ticks for 3d plot.')
        tick_min = data_min
        tick_max = data_max

        cb.set_ticks(numpy.linspace(0,1,11))
        ticklabels_num = numpy.linspace(tick_min,tick_max,11)
        ticklabels = []
        for i, value in enumerate(ticklabels_num):
            ticklabels.append('%.2f'%(value))
        cb.set_ticklabels(ticklabels)

    # applying all ploting options.
    addfieldvalue(options,'colorbaraxis',cb)
    plotApplyOptions(ax,options)
    return varargout(ax,fig)
    # }}}
def plotFlopyTransient(mf,obj,**kwargs): # {{{

    # check data type
    if (not isinstance(obj,flopy.utils.binaryfile.HeadFile)) and (not isinstance(obj,flopy.utils.binaryfile.UcnFile)): 
        raise TypeError('Data(=%s) type should be flopy.utils.binaryfile.UncFile or flopy.utils.binaryfile.HeadFile.'%(type(obj)))

    # get options.
    options  = dictoptions(kwargs)
    debug    = getfieldvalue(options,'debug',False)
    layers   = getfieldvalue(options,'layer',0) # layer=0 for top layer
    ax       = getfieldvalue(options,'ax',[])
    mfPrint('check axes',debug=debug)
    mfPrint(ax,debug=debug)
    mfPrint(type(ax),debug=debug)
    fontsize = getfieldvalue(options,'fontsize',8)
    xlabel   = getfieldvalue(options,'xlabel','x (m)')
    ylabel   = getfieldvalue(options,'ylabel','y (m)')
    title    = getfieldvalue(options,'title',[])
    figno    = getfieldvalue(options,'figno',1)
    outputname = getfieldvalue(options,'output','./temp.gif')

    # get time
    times = obj.get_times()

    # get grid
    xg,yg = flopyGetXyGrid(mf,center=2)

    # find cmax
    cmax = 0
    cmin = 0
    for t in times:
        a = obj.get_data(totim=t)
        cmax_temp = np.amax(a)
        cmin_temp = np.amin(a)
        if cmax < cmax_temp:
            cmax = np.copy(cmax_temp)
        if cmin > cmin_temp:
            cmin = np.copy(cmin_temp)

    # plot animation
    fig = plt.figure(figno); figno = figno +1
    ax = fig.add_subplot()
    temp = obj.get_data(totim=times[0])
    cax = ax.pcolor(xg,yg,temp[0,:,:],cmap='RdBu_r',vmax=cmax,vmin=cmin)
    ax.set_aspect('equal')
    fig.colorbar(cax)
    def update_ani(i,ax,cax,obj,times):
        print('processing = %.2f'%(i/len(times)*100),end='\r')
        temp = obj.get_data(totim=times[i])
        cax.set_array(temp[0,:-1,:-1].flatten())
        ax.set_title('time %f days'%(times[i]))

    ani = animation.FuncAnimation(fig, update_ani,frames=range(len(times)),fargs=(ax,cax,obj,times))
    ani.save(outputname, writer=animation.PillowWriter(fps=20))
# }}}
def plotFlopyTransientValue(mf,obj,**kwargs):# {{{
    '''
    Explain
     plot transient point values 

    Usage
     plotFlopyTransientValue(mf,headobj,index=[nlays, nrows, ncols])
    '''

    # get options
    options  = dictoptions(kwargs)
    if 'index' in options.keys():
        index = options['index']
        lays = index[0]
        rows = index[1]
        cols = index[2]
    elif 'xyz' in options.keys(): 
        xyz = options['xyz']
        cols, rows, lays = flopyXyToIndex(mf,xyz[:,0],xyz[:,1],xyz[:,2]) 
    else:
        raise Exception('check usage of plotFlopyTransientValue')

    # get time and distribution
    ts = obj.get_ts((lays, rows, cols))
    times = ts[:,0]
    data  = ts[:,1]

    # plot time series data at specific point
    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)
    ax.plot(times,data)

    # }}}
def plotApplyOptions(ax, options): # {{{
    '''
    Explain
     Apply all figure options to axes.

    Usage
    '''
    title    = getfieldvalue(options,'title',[])
    fontsize = getfieldvalue(options,'fontsize',8)
    tightsubplot = getfieldvalue(options,'tightsubplot',1)
    colorbartitle = getfieldvalue(options,'colorbartitle',None)
    cb      = getfieldvalue(options,'colorbaraxis',None)

    # label specials
    xlabel  = getfieldvalue(options,'xlabel',None)
    ylabel  = getfieldvalue(options,'ylabel',None)
    xlim = getfieldvalue(options,'xlim',None)
    ylim = getfieldvalue(options,'ylim',None)

    # debugging
    debug   = getfieldvalue(options,'debug',0)

    # which is current figure?
    fig = plt.gcf()

    # set title
    if title:
        ax.set_title(title,fontsize=fontsize)

    mfPrint('set axes ratio',debug=debug)
    ax.set_aspect('auto')

    # set colorbartitle
    if cb:
        if colorbartitle:
            cb.set_label(colorbartitle)


    # set tight_layout()
    if tightsubplot:
        fig.set_tight_layout(True)

    # set x,y axes labels.
    plt.xticks(fontsize=fontsize,axes=ax)
    plt.yticks(fontsize=fontsize,axes=ax)

    if not xlabel:
        ax.set_xlabel('x (m)',fontsize=fontsize)
    else:
        ax.set_xlabel(xlabel,fontsize=fontsize)
    if not ylabel:
        ax.set_ylabel('y (m)',fontsize=fontsize)
    else:
        ax.set_ylabel(ylabel,fontsize=fontsize)

    # set x,y limit
    if xlim:
        ax.set_xlim(options['xlim'])
    if ylim:
        ax.set_ylim(options['ylim'])
# }}}

# check modflow/mt3d path
def check_mf_path(package='mf2005'): # {{{
    if sys.platform == "linux":
        if package == 'mf2005':
            exe_name = 'mf2005'
        elif package == 'mfnwt':
            exe_name = 'mfnwt'

        # check mf name
        if shutil.which(exe_name) == None:
            raise Exception('we cannot find specific location of %s'%(exe_name))

    else: # WINDOW system.
        if package == 'mf2005':
            exe_name = 'mf2005.exe'
        elif package == 'mfnwt':
            exe_name = 'MODFLOW-NWT_64.exe'

        # check mf name
        if shutil.which(exe_name) == None:
            print('we cannot find specific location of %s'%(exe_name))
            # search file at "./bin" directory
            if os.path.isfile('./bin/'+exe_name):
                exe_name ='./bin/'+exe_name
            else:
                print('Download %s package from remote server and install package at current "./bin/".'%(exe_name_mf))
                download_usgs_package(package=package)
                exe_name = './bin/'+exe_name

    return exe_name
    # }}}
def check_mt3d_path(package='mt3dusgs'): # {{{
    if sys.platform == "linux":
        if package == 'mt3dusgs':
            exe_name = 'mt3dusgs'

        # check mf name
        if shutil.which(exe_name) == None:
            raise Exception('we cannot find specific location of %s'%(exe_name))

    else: # WINDOW system.
        if package == 'mt3dusgs':
            exe_name = 'mt3d-usgs_1.1.0_64.exe'

        # check mf name
        if shutil.which(exe_name) == None:
            print('we cannot find specific location of %s'%(exe_name))

            # search file at "./bin" directory
            if os.path.isfile('./bin/'+exe_name):
                exe_name='./bin/'+exe_name
            else:
                print('Download %s package from remote server and install package at current "./bin/".'%(exe_name_mf))
                exe_name='./bin/'+exe_name
                download_usgs_package(package=package)

    return exe_name
    # }}}
def download_package(package='mf2005',install_prefix='./bin/'): # {{{
    '''
    Explain
     download precompiled binary file from remote server

    Usage 
    # install usgs package at "./bin" directory
     download_mf(package='mf2005')
    '''
    debug=1

    # check install prefix
    if not os.path.isdir(install_prefix):
        os.mkdir(install_prefix)

    if package == 'mf2005': # {{{
        prefix = 'MF2005.1_12'

        url = 'https://water.usgs.gov/water-resources/software/MODFLOW-2005/MF2005.1_12.zip'
        print_('download package: %s from USGS homepage'%(package),debug=debug)
        print_('url = %s'%(url),debug=debug)
        wget.download(url)

        # uznip files at "./bin" directory
        shutil.unpack_archive(prefix + '.zip',install_prefix)

        # move all binary files to './bin/'
        filelists = glob.glob(install_prefix + '/' + prefix + '/bin/*.exe')
        for filename in filelists:
            print_('move %s to %s'%(filename,install_prefix),debug=debug)

            # remove file
            if os.path.isfile(install_prefix+'/'+filename.split('/')[-1]):
                os.remove(install_prefix+'/'+filename.split('/')[-1])
            shutil.move(filename,install_prefix)
        # }}}
    if package == 'mt3dusgs': # {{{
        prefix = 'mt3dusgs1.1.0'

        url='https://water.usgs.gov/water-resources/software/MT3D-USGS/mt3dusgs1.1.0.zip'
        print_('download package: %s from USGS homepage'%(package),debug=debug)
        print_('url = %s'%(url),debug=debug)
        wget.download(url)

        # uznip files at "./bin" directory
        shutil.unpack_archive(prefix + '.zip',install_prefix)

        # move all binary files to './bin/'
        filelists = glob.glob(install_prefix + '/' + prefix + '/bin/*.exe')
        for filename in filelists:
            print_('move %s to %s'%(filename,install_prefix),debug=debug)

            # remove file at ./bin/
            if os.path.isfile(install_prefix+'/'+filename.split('/')[-1]):
                os.remove(install_prefix+'/'+filename.split('/')[-1])
            shutil.move(filename,install_prefix)
        # }}}

    # }}}

# Geometry options
def PointsInPolygon(shpfile, x, y): # {{{
    '''
    Explain
     find points location in polygon.

    Usage
        >>> pos = PointsInPolygon(shpfile,x,y)
    '''

    # load polygons from shpfile.
    polygons = geopandas.read_file(shpfile)

    # find points in polygons.
    s = np.shape(x)
    pos = np.zeros((s[0],s[1]))
    for i in range(s[0]):
        for j in range(s[1]):
            p = shapely.geometry.Point(x[i,j],y[i,j])
            pos[i,j] = polygons.contains(p)

    return pos
# }}}
def geopandasSwapXy(x): # {{{
    '''
    Explain 
     swap x,y coordinates of geopandas. (x,y) - > (y,x)

    Ref
     https://gis.stackexchange.com/questions/291247/interchange-y-x-to-x-y-with-geopandas-python-or-qgis
    '''
    coords = list(x.coords)
    coords = [Point(t[1], t[0]) for t in coords] #Swap each coordinate using list comprehension and create Points
    return LineString(coords)
# }}}

# MT3D module.
class Mt3dMas:
    '''
        Explain
         Default data set for reading "MT3D001.MAS".
    '''
    def __init__(self): # {{{
        self.time        = []
        self.total_in    = []
        self.total_out   = []
        self.sources     = []
        self.sinks       = []
        self.net_mass    = []
        self.total_mass  = []
        self.discrepancy = []
        # }}}
def ReadMasFile(filename): # {{{
    '''
        Explain
         import data from "MT3D001.MAS".

        Usage
        # read data file
        >>> data = ReadMasFile('MT3D001.MAS')

        #show plot total mass change
        >>> plt.plot(time, total_mass)
        >>> plt.show()
    '''
    # construct structure
    data = Mt3dMas

    # initialize each variables
    data.time        = []
    data.total_in    = []
    data.total_out   = []
    data.sources     = []
    data.sinks       = []
    data.net_mass    = []
    data.total_mass  = []
    data.discrepancy = []

    # open file 
    fid = open(filename,'r')
    
    cn = 0
    while True:
        cn = cn + 1
        line = fid.readline()
        if not line: break

        # get data more than 2nd lines. This line contains information of budgets.
        if cn > 2:

            # split text line with 'blank(or space)'.
            line_com = line.split()
            data.time        .append(float(line_com[0]))
            data.total_in    .append(float(line_com[1]))
            data.total_out   .append(float(line_com[2]))
            data.sources     .append(float(line_com[3]))
            data.sinks       .append(float(line_com[4]))
            data.net_mass    .append(float(line_com[5]))
            data.total_mass  .append(float(line_com[6]))
            data.discrepancy .append(float(line_com[7]))

    # close file 
    fid.close()

    # return result.
    return data
    # }}}

# About modflow 96 packages.
def mf96ReadBasFloat(fid,ncol,nrow,fileunit): # {{{
    data = np.zeros((nrow*ncol,),dtype=float)
    cn   = 0
    line = fid.readline().split()
    if line[0] == "0":
        print(line)
        data[:] = float(line[1])
    elif line[0] == str(fileunit):
        while True:
            line = fid.readline().split()
            for i in range(len(line)):
                data[cn] = float(line[i])
                cn = cn+1

            if cn == nrow*ncol:
                break
    return data
# }}}
def mf96ReadBasInt(fid,ncol,nrow,fileunit): # {{{
    data = np.zeros((nrow*ncol,),dtype=int)
    cn   = 0
    line = fid.readline().split()
    if line[0] == "0":
        print(line)
        data[:] = int(line[1])
    elif line[0] == str(fileunit):
        while True:
            line = fid.readline().split()
            for i in range(len(line)):
                data[cn] = int(line[i])
                cn = cn+1

            if cn == nrow*ncol:
                break
    return data
# }}}
def mf96ReadRchFloat(fid,ncol,nrow,fileunit,debug=False): # {{{
    data = np.zeros((nrow*ncol,),dtype=float)
    cn   = 0
    line = fid.readline().split()
    mfPrint(line,debug=debug)
    if line[0] == "0":
        print(line)
        data[:] = float(line[1])
    elif line[0] == str(fileunit):
        while True:
            line = fid.readline().split()
            for i in range(len(line)):
                data[cn] = float(line[i])
                cn = cn+1

            if cn == nrow*ncol:
                break
    return data
# }}}

# visual modflow package: mf96
def mf96LoadNam(filename=[],debug=False):# {{{
    # check filename
    if not filename:
        raise Exception('ERROR: mf96LoadNam requires input file.')

    # read all lines
    with open(filename,'r') as fid:
        lines = fid.readlines()
    
    packages={'bas':[],'bcf':[],'wel':[],'drn':[],'riv':[],'rch':[]}
    print(packages.keys())
    for i, line in enumerate(lines):
        package = line.split()
        if package[0].lower() in packages.keys():
            mfPrint('find package = %s'%(package[0]),debug=1)
            packages[package[0].lower()] = [int(package[1]), package[2]]

    return packages
# }}}
def mf96LoadBas(filename=[],fileunit=[],debug=False): # {{{
    if not filename:
        raise Exception('ERROR: mf96LoadBas requires input filename.')
    mfPrint(filename,debug=debug)

    # update filename and fileunit
    filename_bas = filename
    fileunit_bas = fileunit

    # initialize bas_package.
    package_bas={'nlay':[],'nrow':[],'ncol':[],'nper':[],'itmuni':[],
            'ibound':[],'shead':[],'tsmult':[],'istrt':[],'iapart':[],
            'hnoflow':[],'perlen':[],'nstp':[]}

    # read BAS package.
    with open(filename_bas,'r') as fid: # {{{
        # 1 heading
        line = fid.readline()
        # 2 heading
        line = fid.readline()
        # 3 NLAY NROW NCOL NPER ITMUNI
        data = fid.readline().split()
        nlay   = int(data[0])
        nrow   = int(data[1])
        ncol   = int(data[2])
        nper   = int(data[3])
        itmuni = int(data[4]) # time scale 0 - undefined, 1-seconds, 2-minutes, 3-hours, 4-days, 5-years

        # initialize variables.
        ibound = np.zeros((nlay,nrow*ncol),dtype=int)
        shead  = np.zeros((nlay,nrow*ncol),dtype=float)
        perlen = np.zeros((nper,1),dtype=float)
        nstp   = np.zeros((nper,1),dtype=int)
        tsmult = np.zeros((nper,1),dtype=float)

        # 4. options
        data = fid.readline().split()

        # 5. IAPART ISTRT
        mfPrint('   5. IAPART ISTRT',debug=debug)
        line = fid.readline().split()
        iapart = int(line[0]) # iapart = 0 or != 0
        istrt  = int(line[1]) # 0 - initial head not kept, 1 - inital head are kept.

        # 6. IBOUND(NCOL,NROW) or (NCOL,NLAY)
        mfPrint('   6. IBOUND(ncol, nrow)',debug=debug)
        for i in range(nlay):
            ibound[i,:] = mf96ReadBasInt(fid, ncol, nrow, fileunit_bas)

        # 7. HNOFLO
        mfPrint('   7. HNOFLOW(ncol, nrow)',debug=debug)
        line = fid.readline().split()
        Hnoflow = float(line[0])

        # 8. Shead(ncol, nrow) or (ncol, nlay) - starting head
        mfPrint('   8. Shead(ncol, nrow) - starting head',debug=debug)
        for i in range(nlay):
            shead[i,:] = mf96ReadBasFloat(fid, ncol, nrow, fileunit_bas)


        # 9. PERLEN NSTP TSMULT
        mf96Print('   9. perlen, nstp, tsmult - time variables',debug=debug)
        for i in range(nper):
            line = fid.readline().split()
            if not line:
                break
            else:
                perlen = float(line[0])
                nstp   = int(line[1])
                tsmult = float(line[2])
    # }}}

    # save read data to "package_bas"
    package_bas['nlay']=nlay
    package_bas['nrow']=nrow
    package_bas['ncol']=ncol
    package_bas['nper']=nper
    package_bas['itmuni']=itmuni
    package_bas['ibound']=ibound.reshape((nlay,nrow,ncol))
    package_bas['shead']=shead.reshape((nlay,nrow,ncol))
    package_bas['nper']=nper
    package_bas['nstp']=nstp
    package_bas['tsmult']=tsmult
    package_bas['perlen']=perlen
    package_bas['istrt']=istrt
    package_bas['iapart']=iapart
    package_bas['hnoflow']=Hnoflow

    return package_bas
# }}}
def mf96LoadBcf(filename=[],fileunit=[],debug=False,nper=[],nlay=[],ncol=[],nrow=[]):# {{{

    mfPrint('Load BCF package',debug=debug)

    # update filename and fileunit
    filename_bcf = filename
    fileunit_bcf = fileunit

    # initialize variables
    package_bcf = {'delr':[],'delc':[],'sf1':[],'sf2':[],'trans':[],
            'hk':[],'vk':[],'bot':[],'top':[],'wetdry':[]}
    delr = np.zeros((ncol,),dtype=float)
    delc = np.zeros((nrow,),dtype=float)
    sf1   = np.zeros((nlay,ncol*nrow))
    sf2   = np.zeros((nlay,ncol*nrow))
    trans = np.zeros((nlay,ncol*nrow))
    hk    = np.zeros((nlay,ncol*nrow))
    vk    = np.zeros((nlay,ncol*nrow))
    bot   = np.zeros((nlay,ncol*nrow))
    top   = np.zeros((nlay,ncol*nrow))
    wetdry= np.zeros((nlay,ncol*nrow))

    # read BCF package.
    with open(filename_bcf,'r') as fid: # {{{
        # 1.ISS IBCFCB HDRY IWDFLG WETFCT IWETIT IHDWET
        data = fid.readline()
        data = data.split()
        iss    = int(data[0]) # iss = 0 transient / iss = 1 steady state.
        ibcfcb = int(data[1]) # unit number
        hdry   = float(data[2])
        iwdflg = int(data[3]) #
        wetfct = float(data[4]) # a factor that is included in the calculation of the head that is initially established at a cell when it is converted from dry to we
        iwetit = int(data[5]) # iteration interval for attemptiong to wet cells
        ihdwet = int(data[6]) # determines which equation is used to define the initial head at cells that become wet:


        # 2. Ltype(NLAY) - combined code specifying the layer type
        # 0 - harmonic mean
        # 1 - arithmetic mean
        # 2 - logarithmic mean
        # 3 - arithmetic mean of saturated thickness and logarithmic mean of transmissivity
        data = fid.readline().split()
        ltype = np.zeros((nlay,1))
        for i in range(nlay):
            ltype[i] = int(data[i])

        # 3. TRPY(NLAY) -- U1DREL
        data = fid.readline().split()
        if int(data[0]) is fileunit:
            mfPrint('   3. TRPY(NLAY)',debug=debug)
            data = fid.readline().split()
        trpy = np.zeros((nlay,1))
        for i in range(nlay):
            trpy[i] = float(data[i])

        # 4. DELR(NCOL)
        string = '   4. DELR(NCOL)'
        mfPrint(string,debug=debug)
        cn = 0
        data = fid.readline().split()
        while True:
            data = fid.readline().split()
            for i in range(len(data)):
                delr[cn] = float(data[i])
                cn += 1
            if cn == ncol:
                break

        # 5. DELC(NROW)
        string = '   5. DELC(NROW)'
        mfPrint(string,debug=debug)
        # read several lines.
        cn = 0
        data = fid.readline().split()
        while True:
            data = fid.readline().split()
            for i in range(len(data)):
                delc[cn] = float(data[i])
                cn += 1
            if cn == nrow:
                break

        mfPrint('   ncol = %d'%(len(delr)),debug=debug)
        mfPrint('   nrow = %d'%(len(delc)),debug=debug)

        for nl in range(nlay): # {{{
            # 6. Sf1(NCOSL, NROW)# {{{
            if iss == 0: 
                cn = 0
                while True:
                    data = fid.readline().split()
                    if data[0] == str(fileunit) or data[0] == str(0):
                        break
                    else:
                        for i in range(len(data)):
                            sf1[nl,cn] = float(data[i])
                            cn = cn+1
                        if cn == nrow*ncol:
                            break
            # }}}

            # 7. Tran(NCOS, NROW) # {{{
            if ltype[nl] == 0 or ltype[nl] == 2:
                cn = 0
                while True:
                    data = fid.readline().split()
                    if data[0] == str(fileunit) or data[0] == "0":
                        break
                    else:
                        for i in range(len(data)):
                            trans[nl,cn] = float(data[i])
                            cn = cn+1
            # }}}

            # 8. hy(NCOL, NROW) # {{{
            if ltype[nl] == 1 or ltype[nl] == 3:
                string='   8. hy(NCOL, NROW)'
                mfPrint(string,debug=debug)
                cn = 0
                data = fid.readline().split()
                if data[0] == "0":
                    mfPrint(data,debug=debug)
                    hk[nl,:] = float(data[1])*np.ones((nrow*ncol,))
                elif data[0] == str(fileunit):
                    while True:
                        data = fid.readline().split()
                        for i in range(len(data)):
                            hk[nl,cn] = float(data[i])
                            cn = cn+1

                        if cn == nrow*ncol:
                            break
            # }}}

            # 9. Bot(NCOL, NROW) # {{{
            if ltype[nl] == 1 or ltype[nl] == 3:
                string='   9. Bot(NCOL, NROW)'
                mfPrint(string,debug=debug)
                cn = 0
                data = fid.readline().split()
                if data[0] == "0":
                    print(data)
                    bot[nl,:] = float(data[1])*np.ones((nrow*ncol,))
                elif data[0] == str(fileunit):
                    while True:
                        data = fid.readline().split()
                        for i in range(len(data)):
                            bot[nl,cn] = float(data[i])
                            cn = cn+1

                        if cn == nrow*ncol:
                            break
            # }}}

            # 10. Vcont(NCOL, NROW) # {{{
            if nl != nlay-1:
                string='   10. Vcont(NCOL, NROW)'
                mfPrint(string,debug=debug)
                cn = 0
                data = fid.readline().split()
                if data[0] == "0":
                    print(data)
                    vk[nl,:] = float(data[1])*np.ones((nrow*ncol,))
                elif data[0] == str(fileunit):
                    while True:
                        data = fid.readline().split()
                        for i in range(len(data)):
                            vk[nl,cn] = float(data[i])
                            cn = cn+1

                        if cn == nrow*ncol:
                            break
            # }}}

            # 11. Sf2(NCOL, NROW) # {{{
            if iss == 0 and  (ltype[nl] == 1 or ltype[nl] == 3):
                string='11. Sf2(NCOL, NROW)'
                mfPrint(string,debug=debug)
                cn = 0
                data = fid.readline().split()
                if data[0] == "0":
                    print(data)
                    sf2[nl,:] = float(data[1])*np.ones((nrow*ncol,))
                elif data[0] == str(fileunit):
                    while True:
                        data = fid.readline().split()
                        for i in range(len(data)):
                            sf2[nl,cn] = float(data[i])
                            cn = cn+1

                        if cn == nrow*ncol:
                            break
            # }}}

            # 12. Top(NCOL, NROW) # {{{
            if ltype[nl] == 2 or ltype[nl] == 3:
                mfPrint('   12. Top(NCOL, NROW)',debug=debug)
                cn = 0
                data = fid.readline().split()
                if data[0] == "0":
                    print(data)
                    top[nl,:] = float(data[1])*np.ones((nrow*ncol,))
                elif data[0] == str(fileunit):
                    while True:
                        data = fid.readline().split()
                        for i in range(len(data)):
                            top[nl,cn] = float(data[i])
                            cn = cn+1

                        if cn == nrow*ncol:
                            break
            # }}}

            # 13. wetdry(NCOL, NROW) # {{{
            if iwdflg != 0 and (ltype[nl] == 1 or ltype[nl] == 3):
                mfPrint('   13. WetDry(NCOL, NROW)',debug=debug)
                cn = 0
                data = fid.readline().split()
                if data[0] == "0":
                    print(data)
                    wetdry[nl,:] = float(data[1])*np.ones((nrow*ncol,))
                elif data[0] == str(fileunit):
                    while True:
                        data = fid.readline().split()
                        for i in range(len(data)):
                            wetdry[nl,cn] = float(data[i])
                            cn = cn+1

                        if cn == nrow*ncol:
                            break
            # }}}
        # }}}
    # }}}

    # set outputs {{{
    package_bcf['delr']   = delr
    package_bcf['delc']   = delc
    package_bcf['hk']     = hk.reshape((nlay,nrow,ncol))
    package_bcf['sf1']    = sf1.reshape((nlay,nrow,ncol))
    package_bcf['sf2']    = sf2.reshape((nlay,nrow,ncol))
    package_bcf['trans']  = trans.reshape((nlay,nrow,ncol))
    package_bcf['hk']     = hk.reshape((nlay,nrow,ncol))
    package_bcf['vk']     = vk.reshape((nlay,nrow,ncol))
    package_bcf['bot']    = bot.reshape((nlay,nrow,ncol))
    package_bcf['top']    = top.reshape((nlay,nrow,ncol))
    package_bcf['wetdry'] = wetdry.reshape((nlay,nrow,ncol))
    # }}}

    # show results
    return package_bcf
# }}}
def mf96LoadWel(filename=[],fileunit=[],debug=False,nper=[],nlay=[],ncol=[],nrow=[]):# {{{
    
    mfPrint('Load WEL Package')

    # update filename and fileunit
    filename_wel = filename
    fileunit_wel = fileunit

    # initialize variables.
    stress_period = {}

    # read WEL package.
    with open(filename_wel,'r') as fid: # {{{
        mfPrint('   1. MXWELL IWELCB [Option]',debug=debug)
        line = fid.readline().split()
        mxwell  = int(line[0])
        iwelcb  = int(line[1])

        for i in range(nper):
            mfPrint('   2. ITMP - number of wells active during stress period',debug=debug)
            line = fid.readline().split()
            nwel = int(line[0])
            mfPrint('   3. layer, row, column, Q, [xyz]',debug=debug)
            wel_sp = []
            for j in range(nwel):
                line = fid.readline().split()
                if len(line) == 4:
                    wel_sp.append([int(line[0]), int(line[1]), int(line[2]), float(line[3])])

            stress_period.update({i:wel_sp})
    # }}}

    # show results
    #mfPrint(stress_period,debug=debug)
    return stress_period
# }}}
def mf96LoadDrn(filename=[],fileunit=[],debug=False,nper=[],nlay=[],ncol=[],nrow=[]):# {{{
    
    mfPrint('Load DRN Package')

    # update filename and fileunit
    filename_drn = filename
    fileunit_drn = fileunit

    # initialize variables.
    stress_period = {}

    # read DRN package.
    with open(filename_drn,'r') as fid: # {{{
        mfPrint('   1. MXDRN IDRNCB [Option]',debug=debug)
        line = fid.readline().split()
        mxdrn  = int(line[0])
        idrncb = int(line[1])

        stress_period = {}
        for i in range(nper):
            mfPrint('   2. ITMP - number of wells active during stress period',debug=debug)
            line = fid.readline().split()
            ndrn = int(line[0])

            mfPrint('   3. layer, row, column, elevation, cond [xyz]',debug=debug)
            drn_sp = []
            for j in range(ndrn):
                line = fid.readline().split()
                if len(line) == 5:
                    drn_sp.append([int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4])])

            stress_period.update({i:drn_sp})
    # }}}

    # show results
    #mfPrint(stress_period,debug=debug)
    return stress_period
# }}}
def mf96LoadRiv(filename=[],fileunit=[],debug=False,nper=[],nlay=[],ncol=[],nrow=[]):# {{{
    mfPrint('Load RIV Package')

    # update filename and fileunit
    filename_riv = filename
    fileunit_riv = fileunit

    # read RIV package.
    with open(filename_riv,'r') as fid: # {{{
        mfPrint('   1. MXRIV IRIVCB [Option]',debug=debug)
        line = fid.readline().split()
        mxriv  = int(line[0])
        irivcb = int(line[1])

        stress_period = {}
        for i in range(nper):
            mfPrint('   2. ITMP - number of wells active during stress period',debug=debug)
            line = fid.readline().split()
            nriv = int(line[0])

            mfPrint('   3. layer, row, column, stage, cond, rbot [xyz]',debug=debug)
            riv_sp = []
            for j in range(nriv):
                line = fid.readline().split()
                if len(line) == 6:
                    riv_sp.append([int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4]), float(line[5])])

            stress_period.update({i:riv_sp})
    # }}}

    # return output.
    return stress_period
# }}}
def mf96LoadRch(filename=[],fileunit=[],debug=False,nper=[],nlay=[],ncol=[],nrow=[]):# {{{
    mfPrint('Load RCH Package')

    # update filename and fileunit
    filename_rch = filename
    fileunit_rch = fileunit

    mfPrint('   file unit = %d'%(fileunit),debug=debug)

    # initialize variables.
    package_rch={'nrchop':[],'rech':{},'irch':{}}

    # read RCH package.
    with open(filename_rch,'r') as fid: # {{{
        mfPrint('   1. NRCHOP IRCHCB',debug=debug)
        line = fid.readline().split()
        nrchop = int(line[0]) # recharge option. 1-top grid layer, 2-vertical distribution, 3-higest active cell in each vertical column
        irchcb = int(line[1])

        for i in range(nper):
            mfPrint('   2. INRECH INIRCH',debug=debug)
            line = fid.readline().split()
            inrech = int(line[0]) # RECH read flag
            if nrchop == 2:
                inirch = int(line[1]) # IRCH read flag

            mfPrint('   3. RECH(ncol, nrow)',debug=debug)
            rech = mf96ReadRchFloat(fid,ncol,nrow,fileunit=fileunit_rch)
            rech = rech.reshape((nrow,ncol))
            package_rch['rech'].update({i:rech})
    # }}}

    # set outputs.
    package_rch['nrchop'] = nrchop

    # return output.
    return package_rch
# }}}

# visual modflow package: mt3dxx
def vmfLoadAd3(filename,debug=True):# {{{
    raise Exception('this function is not supported yet.')
# }}}

def mf96ToFlopy(mf,bas=[],bcf=[],wel=[],drn=[],riv=[],rch=[]): # {{{
    '''
    Explain
     Save modflow96 to Flopy format.

    Usage
     mf96ToFlopy(mf,bas=bas,bcf=[],wel=[],drn=[],riv=[],rch=[]):
    '''

    flopy.modflow.ModflowDis(mf,nlay=bas['nlay'],nrow=bas['nrow'],ncol=bas['ncol'],
            itmuni=bas['itmuni'],
            nper=bas['nper'],nstp=bas['nstp'],perlen=bas['perlen'],tsmult=bas['tsmult'],
            steady=steady,
            delc=bcf['delc'],delr=bcf['delr'],top=bcf['top'][0,:,:],botm=bcf['bot'])

    flopy.modflow.ModflowBas(mf,ibound=bas['ibound'],strt=bas['shead'])

    flopy.modflow.ModflowLpf(mf,hk=bcf['hk'],vka=bcf['vk'])

    flopy.modflow.ModflowRch(mf,nrchop=rch['nrchop'],
            rech=rch['rech'])

    # check wel package
    # change 1 indices to 0 indices.
    for i in range(len(wel)):
        for j in range(int(np.shape(wel[i])[0])):
            wel[i][j][0] = wel[i][j][0]-1
            wel[i][j][1] = wel[i][j][1]-1
            wel[i][j][2] = wel[i][j][2]-1
    flopy.modflow.ModflowWel(mf,stress_period_data=wel)

    for i in range(len(drn)):
        for j in range(int(np.shape(drn[i])[0])):
            drn[i][j][0] = drn[i][j][0]-1
            drn[i][j][1] = drn[i][j][1]-1
            drn[i][j][2] = drn[i][j][2]-1
    flopy.modflow.ModflowDrn(mf,stress_period_data=drn)

    for i in range(len(riv)):
        for j in range(int(np.shape(riv[i])[0])):
            riv[i][j][0] = riv[i][j][0]-1
            riv[i][j][1] = riv[i][j][1]-1
            riv[i][j][2] = riv[i][j][2]-1
    flopy.modflow.ModflowRiv(mf,stress_period_data=riv)

    # set OC package.
    stress_period_data = {}
    for kper in range(nper):
        for kstp in range(nstp[kper]):
            stress_period_data[(kper,kstp)] = [
                    "save head","save drawdown","save budget","print head"
                    ]
    flopy.modflow.ModflowOc(mf,stress_period_data=stress_period_data,compact=True)
# }}}

# generate mesh
def Hmultiplier(x,dx,multipiler,dx_max,dx_min): # {{{
    output = dx*multipiler

    if abs(output) < dx_min:
        output = dx_min
    if abs(output) > dx_max:
        output = dx_max

    # output
    return output
    # }}}
def grid1d(xmin,xmax,dxmax,dxmin=[],fixpoint=[],multipiler=1.2): # {{{

    if np.any(fixpoint):
        # check fixpoint type
        if isinstance(fixpoint,list):
            fixpoint = np.array(fixpoint)

        if not np.any(dxmin):
            dxmin = dxmax/10

        midpoint = (fixpoint[:-1]+fixpoint[1:])/2
        bcpoint  = np.zeros((len(fixpoint),2),dtype=float)
        for i, x in enumerate(fixpoint):
            if i == 0:
                bcpoint[i,0] = xmin+dxmax/2
                bcpoint[i,1] = midpoint[i]-dxmin/2
            elif i == len(fixpoint)-1:
                bcpoint[i,0] = midpoint[i-1]+dxmin/2
                bcpoint[i,1] = xmax-dxmax/2
            else:
                bcpoint[i,0] = midpoint[i-1]+dxmax/2
                bcpoint[i,1] = midpoint[i]-dxmax/2

    else:
        bcpoint  = np.zeros((1,2),dtype=float)
        fixpoint = [(xmin+xmax)/2]
        bcpoint[0,0] = xmin+dxmax/2
        bcpoint[0,1] = xmax-dxmax/2
        if not np.any(dxmin):
            dxmin = dxmax

    coordglobal = [] # global coordinates
    for i in range(np.shape(bcpoint)[0]):
        #print('%f   (%f, %f)'%(fixpoint[i],bcpoint[i,1],bcpoint[i,1]))
        for j in range(2):
            if j == 0:
                a = bcpoint[i,j]
                b = fixpoint[i]-dxmin/2
            else:
                a = fixpoint[i]+dxmin/2
                b = bcpoint[i,j]
            print('grid gen (%f,%f)'%(a,b))

            coord = [a]
            dx = dxmin
            while coord[-1] < b:
                x = coord[-1]
                dx = Hmultiplier(x,dx,multipiler,dxmax,dxmin)
                xnew = x + dx
                coord.append(xnew)

            if (coord[0]-b) > (b-coord[-1]):
                coord = coord[:-1]

            coord_old = coord
            kappa = (b-coord[-2])/(coord[-1]-coord[-2])
            coord = [a]
            dx = dxmin
            for k in range(0,len(coord_old)-1):
                x = coord_old[k]
                dx = Hmultiplier(x,dx,multipiler,dxmax,dxmin)
                xnew = x + kappa*dx
                coord.append(xnew)

            if j == 0:
                dx_ = np.flip(np.diff(coord))
                nx = len(dx_)
                dx_ = np.matmul(np.tril(np.ones((nx,nx),dtype=float),0),dx_)
                coord = dx_+a
                print(dx_)
                coordglobal.extend(coord)
            else:
                coordglobal.extend(coord)

    # set output
    return coordglobal
    # }}}
