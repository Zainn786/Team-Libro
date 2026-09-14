"""Image-only recognition; no Gazebo entity names, poses, seeds or layout access."""
from pathlib import Path
import cv2
import numpy as np

COLOURS = ('red', 'blue', 'green', 'yellow')


def glyph(mask):
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    crop = mask[y:y+h, x:x+w]
    scale = min(28/w, 44/h)
    resized = cv2.resize(crop, (max(1, round(w*scale)), max(1, round(h*scale))))
    out = np.zeros((48, 32), np.uint8)
    y0, x0 = (48-resized.shape[0])//2, (32-resized.shape[1])//2
    out[y0:y0+resized.shape[0], x0:x0+resized.shape[1]] = resized
    return out


class ShelfVision:
    def __init__(self, template_dir):
        self.templates = {}
        for number in range(1, 6):
            image = cv2.imread(str(Path(template_dir)/f'{number}.png'), 0)
            if image is None:
                raise ValueError(f'Missing digit template {number}')
            self.templates[number] = glyph(cv2.inRange(image, 0, 100))

    def numbers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 65)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (8 <= h <= 140 and 2 <= w <= 120 and .12 < w/h < 1.2):
                continue
            sample = glyph(mask[y:y+h, x:x+w])
            scores = sorted([(float(cv2.matchTemplate(sample, template,
                             cv2.TM_CCOEFF_NORMED)[0, 0]), number)
                             for number, template in self.templates.items()], reverse=True)
            if scores[0][0] >= .35 and scores[0][0]-scores[1][0] >= .04:
                results.append({'number': scores[0][1], 'confidence': scores[0][0],
                                'bbox': [x,y,w,h]})
        return results

    def books(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ranges = {'red': [(0,10),(170,179)], 'yellow': [(20,38)],
                  'green': [(40,85)], 'blue': [(95,135)]}
        results = []
        for colour, bands in ranges.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in bands:
                mask |= cv2.inRange(hsv, (lo,100,55), (hi,255,255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                if h < 10 or w < 2 or cv2.contourArea(contour) < 20 or not .03 < w/h < 1.2:
                    continue
                results.append({'colour': colour, 'bbox': [x,y,w,h]})
        return results

    def bins(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, (0, 100, 45), (12, 255, 255)) |
                cv2.inRange(hsv, (168, 100, 45), (179, 255, 255)))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        image_area = image.shape[0] * image.shape[1]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if h < 18 or w < 30 or area < max(250, image_area * .002):
                continue
            if w / h < 1.25 or area / (w * h) < .35:
                continue
            results.append({'colour': 'red', 'bbox': [x, y, w, h],
                            'confidence': float(min(1., area / max(1., w * h)))})
        return results


def robust_depth(depth, bbox):
    x,y,w,h = bbox
    patch = depth[y+h//4:y+max(h//4+1,3*h//4), x+w//4:x+max(w//4+1,3*w//4)]
    values = patch[np.isfinite(patch) & (patch > .2) & (patch < 8.)]
    if values.size < 3:
        return None
    median = float(np.median(values))
    if float(np.quantile(values,.9)-np.quantile(values,.1)) > .20:
        return None
    return median
