import math
import numpy as np

class Node:
    def __init__(self, parent, position, isTip, ctx, settings, color=None):
        self.parent = parent       # reference to parent node, necessary for vein thickening later
        self.children = []
        self.position = position   # {vec2} of this node's position
        self.isTip = isTip         # {boolean}
        self.ctx = ctx             # global canvas context for drawing
        self.settings = settings
        self.color = color         # color, usually passed down from parent

        self.influencedBy = []     # references to all Attractors influencing this node each frame
        self.killAttractors = []   #
        self.influencedBy_size = 0
        self.thickness = 1       # thickness - this is increased during vein thickening process

    def draw(self):
        if self.parent is not None:
            # Smoothly ramp up opacity based on vein thickness
            if self.settings['EnableOpacityBlending']:
                self.ctx.globalAlpha = self.thickness / 3 + .2

            # "Lines" render mode
            if self.settings['RenderMode'] == 'Lines':
                self.ctx.beginPath()
                self.ctx.moveTo(self.position.x, self.position.y)
                self.ctx.lineTo(self.parent.position.x, self.parent.position.y)

                if self.isTip and self.settings['ShowTips']:
                    self.ctx.strokeStyle = self.settings['Colors']['TipColor']
                    self.ctx.lineWidth = self.settings['TipThickness']
                else:
                    if self.color is not None:
                        self.ctx.strokeStyle = self.color
                    else:
                        self.ctx.strokeStyle = self.settings['Colors']['BranchColor']

                    self.ctx.lineWidth = self.settings['BranchThickness'] + self.thickness

                self.ctx.stroke()
                self.ctx.lineWidth = 1

            # "Dots" render mode
            elif self.settings['RenderMode'] == 'Dots':
                self.ctx.beginPath()
                self.ctx.ellipse(
                    self.position.x,
                    self.position.y,
                    1 + self.thickness / 2,
                    1 + self.thickness / 2,
                    0,
                    0,
                    math.pi * 2
                )

                # Change color or "tip" nodes
                if self.isTip and self.settings['ShowTips']:
                    self.ctx.fillStyle = self.settings['Colors']['TipColor']
                else:
                    self.ctx.fillStyle = self.settings['Colors']['BranchColor']

                self.ctx.fill()


    def tuple_multiply(self, tuple1, len):
        return tuple(x * len for x in tuple1)

    def tuple_sum(self, tuple1, tuple2):
        return tuple(x + y for x, y in zip(tuple1, tuple2))

    # Create a new node in the provided direction and a pre-defined distance (SegmentLength)

    def getNextNode(self, averageAttractorDirection):

        self.isTip = False # 非叶节点
        #displacement = averageAttractorDirection * self.settings['SegmentLength']
        displacement = self.tuple_multiply(averageAttractorDirection, self.settings['SegmentLength'])
        #self.nextPosition = self.position + displacement
        self.nextPosition = self.tuple_sum(self.position, displacement)
        self.nextPosition = (int(self.nextPosition[0]), int(self.nextPosition[1]))
        child = Node(
            self,
            self.nextPosition,
            True,
            self.ctx,
            self.settings,
            self.color
        )
        self.children.append(child)
#       self.nextPosition = self.position.add(averageAttractorDirection.multiply(self.settings['SegmentLength']), True)
        return child
