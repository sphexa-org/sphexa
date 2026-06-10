# add a ParaView source to read H5Part data from SPH-EXA
# works in parallel, detects multiple timesteps
# 
# see documentation https://www.paraview.org/paraview-docs/nightly/python/paraview.util.vtkAlgorithm.html
# 
# Contributed by Jean M. Favre, CSCS
# Tested with ParaView v6.1
#
# This file is loaded from the ParaView GUI with the "Tools->Manage plugins" pull-down menu

from paraview.util.vtkAlgorithm import *
from vtkmodules import vtkCommonDataModel as dm
from vtkmodules.vtkCommonCore import vtkDataArraySelection
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.numpy_interface import algorithms as algs
from vtkmodules.util import vtkConstants
from paraview import print_error, print_warning

try:
  import numpy as np
  import h5py
  _has_deps = True
except ImportError as ie:
  print_error(
        "Missing required Python modules/packages. Algorithms in this module may "
        "not work as expected! \n {0}".format(ie))
  _has_deps = False

def createModifiedCallback(anobject):
    import weakref
    weakref_obj = weakref.ref(anobject)
    anobject = None
    def _markmodified(*args, **kwars):
        o = weakref_obj()
        if o is not None:
            o.Modified()
    return _markmodified

@smproxy.reader(name="H5Reader", label="H5PartPythonReader",
                extensions="h5", file_description="H5Part files from SPH-EXA")
class H5Reader(VTKPythonAlgorithmBase):
    """A reader that reads H5part data from SPH-EXA.
    the data is always treated as a temporal dataset"""
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1, outputType='vtkUnstructuredGrid')
        self._filename = None
        self._ndata = None
        self.timesteps = None
        self._fds = None # H5py file descriptor
        self._arrayselection = vtkDataArraySelection()
        self._arrayselection.AddObserver("ModifiedEvent", createModifiedCallback(self))
        self._vertextype = 2

    def __del__(self):
      if self._fds is not None:
        try:
          self._fds.close()
        except Exception:
          pass
        
    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions="h5", file_description="H5part files from SPH-EXA")
    def SetFileName(self, fname):
        """Specify filename for the file to read."""
        if fname and fname != "None" and self._filename != fname:
            self._filename = fname
            self._ndata = None
            self.timesteps = None
            self.Modified()

    @smproperty.intvector(name="VertexType", number_of_elements="1", default_values="2")
    @smdomain.xml(\
        """<EnumerationDomain name="enum">
          <Entry value="0" text="None"/>
          <Entry value="1" text="Cell Vertex"/>
          <Entry value="2" text="Poly Vertex"/>
        </EnumerationDomain>
        <Documentation>select one out of 3 possibilities to create a cellset</Documentation>
        """)
    def SetVertexType(self, vertextype):
        """Specify the vertex type for cells."""
        if vertextype and self._vertextype != vertextype:
            self._vertextype = vertextype
            self.Modified()
            
    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        return self.timesteps

    @smproperty.dataarrayselection(name="Arrays")
    def GetDataArraySelection(self):
        return self._arrayselection

    def FillOutputPortInformation(self, port, info):
        from vtkmodules.vtkCommonDataModel import vtkDataObject
        if port == 0:
            info.Set(vtkDataObject.DATA_TYPE_NAME(), "vtkUnstructuredGrid")
        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        from vtkmodules.vtkCommonExecutionModel import (
            vtkStreamingDemandDrivenPipeline,
            vtkAlgorithm,
        )
        # tell the pipeline it is a parallel reader
        executive = vtkStreamingDemandDrivenPipeline
        port = outInfo.GetInformationObject(0)
        port.Set(vtkAlgorithm.CAN_HANDLE_PIECE_REQUEST(), 1)
        
        port.Remove(executive.TIME_STEPS())
        port.Remove(executive.TIME_RANGE())
        #print(f'opening file {self._filename}')
        self._fds = h5py.File(self._filename,'r')
        self.timesteps = []
        for i in range(len(self._fds.keys())):
          val = self._fds[f'Step#{i}'].attrs['time'][0]
          self.timesteps.append(val)

        for timestep in self.timesteps:
          port.Append(executive.TIME_STEPS(), timestep)
        port.Append(executive.TIME_RANGE(), self.timesteps[0])
        port.Append(executive.TIME_RANGE(), self.timesteps[-1])
        self._ndata = []
        for aname in self._fds['Step#0'].keys():
          self._arrayselection.AddArray(aname)
          self._ndata.append(aname)
          #print("adding ", aname, " to variable selection")
        return 1
        
    def _get_time_index(self, outInfo):
        executive = self.GetExecutive()
        time_info = outInfo.GetInformationObject(0)
        if time_info.Has(executive.UPDATE_TIME_STEP()) and len(self.timesteps) > 1:
            time = time_info.Get(executive.UPDATE_TIME_STEP())
            for i, t in enumerate(self.timesteps):
                if time <= t:
                    return i
        return 0
        
    def RequestData(self, request, inInfo, outInfoVec):
        # coordinates are read as float32 to save space
        # all other variables are read at their native resolution in the file
        # when run in parallel, the domain will be split equaly among MPI tasks
        from vtkmodules.vtkCommonDataModel import vtkCellArray
        from vtkmodules.vtkCommonCore import vtkAffineIntArray
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline
        if not _has_deps:
          print_error("Required Python module 'h5py' or 'numpy' missing!")
          return 0

        executive = vtkStreamingDemandDrivenPipeline

        outInfo = outInfoVec.GetInformationObject(0)
        piece = outInfo.Get(executive.UPDATE_PIECE_NUMBER())
        npieces = outInfo.Get(executive.UPDATE_NUMBER_OF_PIECES())
        
        output = dsa.WrapDataObject(vtkUnstructuredGrid.GetData(outInfo))
        tsindex = self._get_time_index(outInfoVec)
        
        timedata = self._fds[f'Step#{tsindex}']
        nparticles = timedata['x'].shape[0]
        if npieces == 1:
          nlocalparticles = MyNumber_of_Cells = nparticles
        else:
          nlocalparticles = nparticles // npieces
        if piece < (npieces-1):
          MyNumber_of_Cells = nlocalparticles
        else:
          MyNumber_of_Cells = nparticles - (npieces-1) * nlocalparticles

        begin = piece * nlocalparticles
        end = piece * nlocalparticles + MyNumber_of_Cells
        #print(f'Piece {piece}/{npieces} Grabbing data [{begin}:{end}] from Step#{tsindex}')
        coords_x = timedata['x'][begin:end].astype('f4')
        coords_y = timedata['y'][begin:end].astype('f4')
        coords_z = timedata['z'][begin:end].astype('f4')
        output.Points = algs.make_vector(coords_x, coords_y, coords_z)
        for name in self._ndata:
          if self._arrayselection.ArrayIsEnabled(name):
            dataset = timedata[name][begin:end]
            output.PointData.append(dataset, name)
        if self._vertextype == 1:
          vertex = vtkAffineIntArray()
          vertex.ConstructBackend(1,0)
          vertex.SetNumberOfTuples(MyNumber_of_Cells)
          verts = vtkCellArray()
          verts.AllocateExact(MyNumber_of_Cells, MyNumber_of_Cells) # numCells, connectivitySize
          # offsets array below is automatically generated given the fixed cells size
          verts.SetData(1, vertex)
          output.VTKObject.SetCells(vtkConstants.VTK_VERTEX, verts)
        elif self._vertextype == 2:
          polyVertex = vtkAffineIntArray()
          polyVertex.ConstructBackend(1,0)
          polyVertex.SetNumberOfTuples(MyNumber_of_Cells)
          verts = vtkCellArray()
          verts.AllocateExact(1, MyNumber_of_Cells) # numCells, connectivitySize
          verts.SetData(MyNumber_of_Cells, polyVertex)
          output.VTKObject.SetCells(vtkConstants.VTK_POLY_VERTEX, verts)
        return 1

