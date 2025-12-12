import math

class Attractor:
    def __init__(self,position, ctx, settings=None):
        self.position = position
        self.ctx = ctx
        self.settings = None  # Not initialized
        self.influencingNodes = []
        self.fresh = True
        self.reached = False

    def draw(self):
        if self.settings['ShowAttractionZones']:
            self.ctx.beginPath()
            self.ctx.ellipse(self.position.x, self.position.y, self.settings['AttractionDistance'], self.settings['AttractionDistance'], 0, 0, math.pi * 2)
            self.ctx.fillStyle = self.settings['Colors']['AttractionZoneColor']
            self.ctx.fill()

        # Draw the kill zone
        if self.settings['ShowKillZones']:
            self.ctx.beginPath()
            self.ctx.ellipse(self.position.x, self.position.y, self.settings['KillDistance'], self.settings['KillDistance'], 0, 0, math.pi * 2)
            self.ctx.fillStyle = self.settings['Colors']['KillZoneColor']
            self.ctx.fill()

        # Draw the attractor
        if self.settings['ShoAttractors']:  # Note: typo in original code ("ShoAttractors" instead of "ShowAttractors")
            self.ctx.beginPath()
            self.ctx.ellipse(self.position.x, self.position.y, 1, 1, 0, 0, math.pi * 2)
            self.ctx.fillStyle = self.settings['Colors']['AttractorColor']
            self.ctx.fill()
