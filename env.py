from settings import *
from weather import *

class RoomSimulator(Env):
    """
    ### Parameters

    heating_power: Number of degC the room's temperature goes up by in 15 minutes, if the heating is on, given no heat loss to surroundings (ideal value = 0.5).\n
    loss_coefficient: Given a temperature difference of 10C between inside and outside, the loss coefficient describes the number of degC the room's temperature drops by within 15 minutes. \n
    reward_mech: "V1" uses a reward mechanism with three bands. "V2" uses a reward mechanism with one band (see step module for more details)
    render_mode: "human" 

    ### Description

    This environment simulates a room's internal temperature profile for a Summer's day in the UK. 

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
    of the fixed force the cart is pushed with.

    | Num | Action         |
    |-----|----------------|
    | 0   | Heating is off |
    | 1   | Heating is on  |

    ### Observation Space

    The observation is a `ndarray` with shape `(1,)` with the values corresponding to the following:

    | Num | Observation     | Min                | Max               |
    |-----|-----------------|--------------------|-------------------|
    | 0   | Room Temperature| -20                | 60                |

    ### Rewards

    Since the goal is to keep the room's temperature as close to the setpoint temperature, rewards are based on bands:
    - Reward of `+1` is awarded for every step taken when the room's temperature is within 0.5C of the setpoint. 
    - Reward of '+0.6' is awarded for every step taken when the room's temperature is within 0.5C - 1C of the setpoint.
    - Reward of '+0.3' is awarded for every step taken when the room's temperature is within 1C - 1.5C of the setpoint.
    - Reward of '0' is awarded for every step taken when the room's temperature is more than 1.5C away from the setpoint.

    ### Starting State

    Room temperature is assigned a uniformly random value in `(18, 20)`

    ### Episode End

    The episode ends when the day is over or 96 fifteen minute intervals have elapsed. 
    
    """

    metadata = {
        "render_fps":12
    }
    
    def __init__(self, heating_power, loss_coefficient, reward_mech: Optional[str] = "V1", render_mode: Optional[str] = None):

        # spaces
        self.observation_space = Box(low = -20, high = 60, shape = (1,))
        self.action_space = Discrete(n = 2)

        # parameters
        self.h = heating_power
        self.l = loss_coefficient
        self.reward_mech = reward_mech
        self.reward_mech_list = ["V1","V2"]
        self.num_timesteps = np.arange(0,len(t))
        self.done = None

        # render
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        self.black = (0,0,0)
        self.white = (255,255,255)
        self.blue = (0,0,255)
        self.green = (0,255,0)
        self.ts_colour = self.black
        self.setpoint_colour = self.green

        self.screen_width = 1800
        self.screen_height = 800

        # chart
        self.chart_xoffset = 100
        self.chart_yoffset = 100
        self.chart_width = 1200
        self.chart_height = 600
        self.origin = (140, 660)
        self.xtick = 11.6665 # pixels/15min
        self.ytick = 104/5 # pixels/degC   
        self.min_temp = 15

        # displays
        self.rect_xoffset = 1350
        self.rect_width = 400
        self.recttorect = 50
        self.rect1_height = 150
        self.rect2_height = 200  
        

    def step(self,action):
        
        # update state (indoor temperature)
        self.state += (self.h*action) + (self.l*(self.otemp[self.current_timestep] - self.state))
        self.action = action

        # reward mechanism
        if self.reward_mech in self.reward_mech_list:
            pass
        else:
            gym.logger.warn(
                f"You have specified an invalid reward mechanism. Currently accepted reward mechanisms are {self.reward_mech_list}"
            ) 
            return
        
        if self.reward_mech == 'V1':
            if abs(self.setpoint[int(self.current_timestep)] - self.state) <= 0.5: reward = 1
            elif 0.5 < abs(self.setpoint[int(self.current_timestep)] - self.state) <= 1: reward = 0.6
            elif 1 < abs(self.setpoint[int(self.current_timestep)] - self.state) <= 1.5: reward = 0.3
            else: reward = 0
        else: # V2
            if abs(self.setpoint[int(self.current_timestep)] - self.state) <= 0.5: reward = 1
            else: reward = 0

        # is the day finished?
        if self.current_timestep == len(t) -1:
            self.done = True
        else:
            self.done = False

        # update timestep & score
        self.current_timestep += 1
        if self.current_timestep <= 95: self.ts[self.current_timestep] = self.state
        self.score += reward
        info = {self.score}

        return self.state, reward, self.done, info
            
    def reset(self):
        
        self.state = random.uniform(18,20)
        self.otemp = OutdoorTemp().o_temp_summer
        self.setpoint = SETPOINT_SUMMER
        self.current_timestep = 0
        self.score = 0
        self.ts = np.zeros(len(t))
        self.ts[self.current_timestep] = self.state
    
        return self.state

    def render(self):
        
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                'You can specify the render_mode at initialization e.g. RoomSimulator(...,render_mode= "human")'
            )
            return 

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            )

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                gym.logger.warn(
                    "You have specified an unknown render_mode"
                )
                return

        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # imports
        self.font = pygame.font.Font(None, 30)
        self.font_big = pygame.font.Font(None,40)
        self.heater = pygame.transform.rotozoom(pygame.image.load(join('media','heater.png')).convert_alpha(),0,0.1)
        
        # draw elements
        self.screen.fill(self.white)
        self.draw_chart()
        self.draw_lines()
        self.display_score()
        self.display_heating_info()
        self.display_legend()
        pygame.display.update()
        
        # framerate
        self.clock.tick(self.metadata["render_fps"]) 
        pygame.event.pump() # seems to not crash when I try to close the pygame window if I include this line
        self.pause()

    def close(self):
        import pygame
        
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def draw_chart(self):
        # chart box
        chart_rect = pygame.Rect(self.chart_xoffset,self.chart_yoffset,self.chart_width,self.chart_height)
        pygame.draw.rect(self.screen,self.black,chart_rect,2)
    
        # time labels
        for i in range(5):
            x = self.origin[0] + i * (self.xtick*4*5)
            pygame.draw.line(self.screen, self.black, (x, self.chart_yoffset + self.chart_height), (x, self.chart_yoffset + self.chart_height + 10))
            time_label = self.font.render(f'{i*5:02}:00', True, self.black)
            self.screen.blit(time_label,(x-20, self.chart_yoffset + self.chart_height + 20))

        # temperature labels
        for i in range(6):
            y = self.origin[1] + -i * (self.ytick*5)
            pygame.draw.line(self.screen,self.black,(100,y),(90,y))
            temp_label = self.font.render(f"{self.min_temp+i*5}",True,self.black)
            self.screen.blit(temp_label,(65, y-7.5))
    
        y_axis_label = pygame.transform.rotate(self.font.render('Room Temperature (°C)', True, self.black),90)
        y_axis_label_rect = y_axis_label.get_rect(center = (self.chart_xoffset/2 - 10,self.screen_height/2))
        self.screen.blit(y_axis_label, y_axis_label_rect)

    def draw_lines(self):

        # setpoint
        if self.current_timestep <= len(t) - 1:
            for i in range(self.current_timestep):
                x1 = self.origin[0] + self.xtick*i
                x2 = self.origin[0] + self.xtick*(i+1)
                y1 = self.origin[1] - ((self.setpoint[i]-self.min_temp)*self.ytick)
                y2 = self.origin[1] - ((self.setpoint[i+1]-self.min_temp)*self.ytick)
                pygame.draw.line(self.screen,self.setpoint_colour,(x1,y1),(x2,y2),2)

        # room temp
        if self.current_timestep <= len(t) - 1:
            for i in range(self.current_timestep):
                x1 = self.origin[0] + i*self.xtick
                x2 = self.origin[0] + (i+1)*self.xtick
                y1 = self.origin[1] - ((self.ts[i]-self.min_temp)*self.ytick)
                y2 = self.origin[1] - ((self.ts[i+1]-self.min_temp)*self.ytick)
                pygame.draw.line(self.screen,self.ts_colour,(x1,y1),(x2,y2),2)
    
    def display_score(self):
        score_rect = pygame.Rect(self.rect_xoffset,self.chart_yoffset,self.rect_width,self.rect1_height)
        pygame.draw.rect(self.screen, self.white,score_rect, 2)
        score_text = self.font.render(f"Score: {self.score:.1f}",False,self.black)
        score_text_rect = score_text.get_rect(center = (self.rect_xoffset + self.rect_width/2,self.chart_yoffset + self.rect1_height/2))
        self.screen.blit(score_text, score_text_rect)
    
    def display_heating_info(self):
        rect = pygame.Rect(self.rect_xoffset,self.chart_yoffset+self.rect1_height+self.recttorect,self.rect_width,self.rect2_height)
        heater_rect = self.heater.get_rect(center = (self.rect_xoffset + self.rect_width/2,self.chart_yoffset+self.rect1_height+self.recttorect+self.rect2_height/2))

        if self.action==1: pygame.draw.rect(self.screen,'red',heater_rect)
        pygame.draw.rect(self.screen, self.white,rect, 2)
        self.screen.blit(self.heater,heater_rect)
    
    def display_legend(self):

        lines_xoffset = self.rect_width/2
        setpoint_lines_yoffset = 40

        rect = pygame.Rect(self.rect_xoffset,self.chart_yoffset+self.rect1_height+self.recttorect+self.rect2_height+self.recttorect,self.rect_width,self.rect1_height)
        room_txt = self.font.render("Room",False,self.black)
        room_txt_rect = room_txt.get_rect(center = (self.rect_xoffset + self.rect_width/4,self.chart_height+self.chart_yoffset-self.rect1_height+setpoint_lines_yoffset))
        setpoint_txt = self.font.render("Setpoint",False,self.black)
        setpoint_txt_rect = setpoint_txt.get_rect(center = (self.rect_xoffset + self.rect_width/4,self.chart_height+self.chart_yoffset-setpoint_lines_yoffset))

        pygame.draw.rect(self.screen, self.white,rect, 2)
        pygame.draw.line(self.screen,self.ts_colour,(self.rect_xoffset + lines_xoffset, self.chart_height+self.chart_yoffset-self.rect1_height+setpoint_lines_yoffset), 
                         (self.rect_xoffset + self.rect_width-30,self.chart_height+self.chart_yoffset-self.rect1_height+setpoint_lines_yoffset),2)
        pygame.draw.line(self.screen,self.setpoint_colour,(self.rect_xoffset + lines_xoffset,self.chart_height+self.chart_yoffset-setpoint_lines_yoffset), 
                         (self.rect_xoffset + self.rect_width-30,self.chart_height+self.chart_yoffset-setpoint_lines_yoffset),2)
        self.screen.blit(room_txt,room_txt_rect)
        self.screen.blit(setpoint_txt,setpoint_txt_rect)
    
    def pause(self):
        """
        Pauses the line chart output at the end of the day for X seconds
        """
        if self.current_timestep == len(t) - 1:
            pygame.time.delay(5000)

    