import xml.etree.ElementTree as ET
import numpy as np,struct,re,json,sys
positions={}
from pathlib import Path
from scipy.spatial.transform import Rotation
import argparse
parser=argparse.ArgumentParser(description='Generate conservative collision bounds from official URDF meshes')
parser.add_argument('source_root',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
root=args.source_root
packages={ET.parse(p).getroot().findtext('name'):p.parent for p in root.rglob('package.xml')}
r=ET.parse(root/'erc_description/urdf/tiago_pro.urdf').getroot()
def origin(e):
 t=np.eye(4)
 if e is not None:
  t[:3,3]=[float(x) for x in e.get('xyz','0 0 0').split()]
  t[:3,:3]=Rotation.from_euler('xyz',[float(x) for x in e.get('rpy','0 0 0').split()]).as_matrix()
 return t
transforms={'base_footprint':np.eye(4)};todo=list(r.findall('joint'))
while todo:
 progress=False
 for j in todo[:]:
  parent=j.find('parent').get('link');child=j.find('child').get('link')
  if parent in transforms:
   motion=np.eye(4);value=positions.get(j.get('name'),0.)
   if j.get('type') in ('revolute','continuous'):
    axis=np.array([float(x) for x in j.find('axis').get('xyz').split()]);motion[:3,:3]=Rotation.from_rotvec(axis*value).as_matrix()
   elif j.get('type')=='prismatic':motion[:3,3]=np.array([float(x) for x in j.find('axis').get('xyz').split()])*value
   transforms[child]=transforms[parent]@origin(j.find('origin'))@motion;todo.remove(j);progress=True
 if not progress:raise ValueError('TF cycle')
allpts=[];ranges=[];local_corners={}
for l in r.findall('link'):
 pts=[]
 for c in l.findall('collision'):
  g=c.find('geometry');mesh=g.find('mesh')
  if mesh is not None:
   uri=mesh.get('filename').removeprefix('package://');pkg,rel=uri.split('/',1);data=(packages[pkg]/rel).read_bytes()
   n=struct.unpack('<I',data[80:84])[0]
   if len(data)==84+50*n:
    dt=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')]);v=np.frombuffer(data[84:],dtype=dt)['v'].reshape(-1,3)
   else:v=np.array([[float(x) for x in m] for m in re.findall(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)',data)])
   v=v*np.array([float(x) for x in mesh.get('scale','1 1 1').split()]);lo=v.min(axis=0);hi=v.max(axis=0)
  elif g.find('box') is not None:hi=np.array([float(x) for x in g.find('box').get('size').split()])/2;lo=-hi
  elif g.find('sphere') is not None:hi=np.ones(3)*float(g.find('sphere').get('radius'));lo=-hi
  elif g.find('cylinder') is not None:
   cy=g.find('cylinder');hi=np.array([float(cy.get('radius'))]*2+[float(cy.get('length'))/2]);lo=-hi
  else:continue
  corners=np.array([[x,y,z,1] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
  local=(origin(c.find('origin'))@corners.T).T[:,:3]
  local_corners.setdefault(l.get('name'),[]).extend(local.tolist())
  pts.extend((transforms[l.get('name')]@origin(c.find('origin'))@corners.T).T[:,:3])
 if pts:
  pts=np.array(pts);allpts.extend(pts);ranges.append((l.get('name'),pts.min(axis=0).round(3).tolist(),pts.max(axis=0).round(3).tolist()))
print('full robot bounds',np.min(allpts,axis=0),np.max(allpts,axis=0))
for name,lo,hi in ranges:
 if hi[0]>.7 or abs(lo[1])>.5 or abs(hi[1])>.5:print(name,lo,hi)

import hashlib
args.output.write_text(json.dumps({'urdf_sha256':hashlib.sha256((root/'erc_description/urdf/tiago_pro.urdf').read_bytes()).hexdigest(),'links':local_corners},indent=2))
