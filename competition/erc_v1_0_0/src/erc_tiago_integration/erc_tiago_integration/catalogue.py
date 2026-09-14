"""Fuse independent camera observations; shelf labels never imply coordinates."""
import math
from copy import deepcopy
import numpy as np

COLOURS=('red','blue','green','yellow')


class ObservationCatalogue:
    def __init__(self, required_observations=3, max_spread=.08):
        self.required=required_observations
        self.max_spread=max_spread
        self.samples={}

    def add(self, key, observation):
        point=np.asarray(observation['point'],float)
        if point.shape!=(3,) or not np.isfinite(point).all():
            return False
        samples=self.samples.setdefault(key,[])
        if any(x['stamp_ns']==observation['stamp_ns'] for x in samples):
            return False
        if samples and np.linalg.norm(point-np.median([x['point'] for x in samples],axis=0))>self.max_spread:
            self.samples[key]=[]
            return False
        samples.append(deepcopy(observation))
        del samples[:-8]
        return len(samples)>=self.required

    def confirmed(self):
        result={}
        for key,samples in self.samples.items():
            if len(samples)<self.required:
                continue
            item=deepcopy(samples[-1])
            item['point']=np.median([x['point'] for x in samples],axis=0).tolist()
            item['observation_count']=len(samples)
            result[key]=item
        return result


def shelf_axes(markers, observer):
    """Estimate shelf direction from observed overhead markers in odometry."""
    points=np.array([x['point'][:2] for x in markers.values()])
    if len(points)<3:
        raise ValueError('Need at least three observed markers to determine shelf direction')
    center=points.mean(axis=0)
    _,singular,vh=np.linalg.svd(points-center)
    if singular[0]<.6 or singular[1]>.12:
        raise ValueError('Observed shelf markers do not form a consistent shelf line')
    tangent=vh[0]
    normal=np.array([-tangent[1],tangent[0]])
    if np.dot(center-np.array(observer[:2]),normal)<0:
        normal=-normal
    return tangent,normal


def approach_pose(book, normal, stand_off=1.10):
    if not .9<=stand_off<=2.:
        raise ValueError('Unvalidated travel stand-off')
    normal=np.asarray(normal,float)
    if not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1.)>.001:
        raise ValueError('Invalid shelf normal')
    xy=np.array(book['point'][:2])-stand_off*normal
    return float(xy[0]),float(xy[1]),math.atan2(normal[1],normal[0])


def assign_rows(books):
    if set(books)!=set(COLOURS):
        raise ValueError('A shelf column must contain one observed book of each colour')
    ordered=sorted(books.values(),key=lambda x:-x['point'][2])
    if any(not .20<ordered[i]['point'][2]-ordered[i+1]['point'][2]<.45 for i in range(3)):
        raise ValueError('Observed books do not occupy four distinct, plausible rows')
    return {item['colour']:{**item,'row':i+1} for i,item in enumerate(ordered)}
