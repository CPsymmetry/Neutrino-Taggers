# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:15:58 2023

@author: Taylor c.
"""
import matplotlib.pyplot as plt
import numpy as np

class pconstructor:
    def __init__(self, energy = 0, theta = 0, phi = 0, origin = 0): 
        """
        Constructs a hypothetical particle with defined properties

        Parameters
        ----------
        energy : float, optional
            Particle energy. The default is 0.
        theta : float, optional
            particles angle. The default is 0.
        phi : float, optional
            particles rotational angle. The default is 0.
        origin : float, optional
            origin of particle. The default is 0.

        Returns
        -------
        None.

        """
        self.energy = energy
        self.origin = origin
        self.position = self.origin
        self.theta = theta
        self.phi = phi
        
    def move(self, z):
        """
        Moves the particle to a particular z position
        
        Parameters
        ----------
        z : float
            z position to move the particle to.
        
        Returns
        -------
        self.position : array
            array of positions, x,y,z.
        """
        origin = self.origin[2]
        x = (z-origin)*np.tan(self.theta)*np.sin(self.phi)
        y = (z-origin)*np.tan(self.theta)/np.cos(self.phi)
        
        self.position = [x,y,z]
        
        return self.position
                   
class geometry:
    def __init__(self):
        """
        A place holder class to hold geometry types.
        """
        return None
    
    class circular:
        def __init__(self, info):
            """
            Circular detector geometry
            
            Parameters
            ----------
            Info : dictionary
                {
                range : 'range of detector' [zi, zf] (array, meters)
                pos : 'its offset to the center' (meters (x,y,z=0))
                radius : 'radius of circle' (meters)
                inner_radius : 'inner radius of hole in circle' (meters)
                }
                
                Contains all information in regards to the detector geometry
            """
            if type(info) == dict:
                self.geometry_fufilled = True
                self.range = info['range']
                self.length = self.range[1] - self.range[0]
                self.pos = info['pos']
                self.z = self.range[0]
                self.radius = info['radius']
                self.inner_radius = info['inner_radius']
                return None
        
        def fit(self, pos):
                """
                Checks if position of particle is within the geometry of the detector

                Parameters
                ----------
                pos : array
                    array of particle position (x,y,z).

                Returns
                -------
                bool
                    true if within the detector geometry.
                """
                x = pos[0]
                y = pos[1]
                pmag = np.sqrt(x**2 + y**2)
                if pmag >= self.inner_radius and pmag <= self.radius:
                    return True
                else:
                    return False
        
    class rectangular:
        def __init__(self, info):
            """
            Rectangular detector geometry
            
            Parameters
            ----------
            Info : dictionary
                {
                range : 'range of detector' [zi, zf] (array, meters)
                pos : 'its offset to the center' (meters (x,y,z=0))
                width : 'detector width' (meters)
                height : 'detector height' (meters)
                inner_radius : 'inner radius of hole in circle' (meters)
                }
                
                Contains all information in regards to the detector geometry
            """
            if type(info) == dict:
                self.geometry_fufilled = True
                self.range = info['range']
                self.length = self.range[1] - self.range[0]
                self.pos = info['pos']
                self.z = self.range[0]
                self.width = info['width']
                self.height = info['height']
                self.inner_radius = info['inner_radius']
                return None
        
        def fit(self, pos):
            """
            Checks if position of particle is within the geometry of the detector

            Parameters
            ----------
            pos : array
                array of particle position (x,y,z).

            Returns
            -------
            bool
                true if within the detector geometry.
            """
            x = pos[0]
            y = pos[1]
            pmag = np.sqrt(x**2 + y**2)
            if pmag >= self.inner_radius and np.abs(x) <= self.width/2 and np.abs(y) <= self.height/2:
                return True
            else:
                return False

class na62:
    def __init__(self):
        """
        NA62 Detector geometric acceptance and event selection class.

        """
        self.min_muon_energy = 15
        self.min_neutrino_energy = 5
        self.c = 3*10**8
        self.p_rest = 0.2358
        self.e_rest = 0.258
        self.beta_gamma = 75/.493
        self.gamma = np.sqrt(.493**2+75**2)/.493
        
        self.straw = self.straw()
        self.chod = self.chod()
        self.rich = self.rich()
        self.muv3 = self.muv3()
        self.lkr = self.lkr()
        
    def simulate(self, pnum):
        """
        Simulates pnum number of particles for the na62 detector.

        Parameters
        ----------
        pnum : int
            number of particles to be simulated.

        Returns
        -------
        n : int
            number of successful events.
        list
            list of dictionaries of all data on accepted and total simulated particles.
        """
        tp = []
        ps = [] 
        n = 0
        for i in range(pnum):
            success, et = self.simulate_particle()
            tp.append(et)
            if success:
                n+=1
                ps.append(et)
            
        return n, [ps, tp]
    
    
    def simulate_particle(self):
        """
        Simulates a singular particle going through the detector
        
        Returns
        -------
        boolean :
            Whether the particle is geometrically accepted
        et : dictionary
            contains all information about the particles for use in data analysis.
        """
        #simulate individual particles.
        #Kaon Energy = 75 GeV,  Mass, .494 GeV/c^2
        origin = self.kaon_decay_pos()
        mu_energy, mu_theta, mu_phi, costheta, sintheta = self.muon_et()
        n_energy, n_theta, n_phi = self.neutrino_et(mu_energy, costheta, sintheta, mu_phi)
  
        et = {
              'm_energy': mu_energy,
              'm_theta': mu_theta,
              'm_phi': mu_phi,
              'n_energy': n_energy,
              'n_theta': n_theta,
              'n_phi': n_phi,
              'origin': origin
            }
        
        if n_energy < self.min_neutrino_energy or mu_energy < self.min_muon_energy:
            return False, et
            
        muon = pconstructor(energy = et['m_energy'], theta = et['m_theta'], phi = et['m_phi'], origin = [0,0,et['origin']])
        neutrino = pconstructor(energy = et['n_energy'], theta = et['n_theta'], phi = et['n_phi'], origin = [0,0,et['origin']])

        test1 = self.test_detector(self.straw, muon)
        if test1:
            muon = self.mu_p_kick(muon)
            test2 = self.test_detector(self.rich, muon)
            if test2:        
                test3 = self.test_detector(self.chod, muon)
                if test3:
                    test4 = self.test_detector(self.muv3, muon)
                    if test4:
                        test5 = self.test_detector(self.lkr, neutrino)
                        if test5:
                           return True, et
    
        return False, et
    
    def mu_p_kick(self, mu):
        z = 200
        mu_mass = .1057
        mu_pos = mu.move(z)  
        mu_energy = mu.energy
        mu_theta = mu.theta
        mu_phi = mu.phi
        
        costheta = np.cos(mu_theta)
        sintheta = np.sin(mu_theta)
        
        sinphi = np.sin(mu_phi)
        cosphi = np.cos(mu_phi)
        
        mu_p = mu_energy**2 - mu_mass**2
        a = np.sqrt((mu_p*sintheta*sinphi + .270)**2 + (mu_p*sintheta*cosphi + .027)**2)
        b = mu_p*costheta
        
        c = mu_p*sintheta*sinphi + .270
        d = mu_p*sintheta*cosphi + .027
        
        n_mu_theta = np.arctan(a/b)
        n_mu_phi = np.arctan(c/d)
        
        mu = pconstructor(energy = mu_energy, theta = n_mu_theta, phi = n_mu_phi, origin = mu_pos)
        
        return mu
    
    @staticmethod
    def kaon_decay_pos():
        """
        Creates a random decay position of the kaon dependent on a uniform distribution.
        Actual kaon decay has a poisson distribution but difference is negligable.

        Returns
        -------
        kaon_pos : float
            Position of kaon decay

        """
        kaon_pos = np.random.uniform(102.4, 183.316)
        return kaon_pos
    
    def muon_et(self):
        """
        Calculates the lab energy and angle of the muon
        
        Returns
        -------
        lmu_energy : float
            lab frame energy of the muon
        lmu_theta : float
            lab frame angle of the muon
        phi : TYPE
            rotational angle of the muon
        costheta : float
            cos of theta
        sintheta : float
            sin of theta.

        """
        costheta = np.random.uniform(-1,1)
        phi = np.random.uniform(-np.pi, np.pi)
        ve = np.random.choice([-1,1])
        sintheta = ve*np.sqrt(1 - costheta * costheta)
        
        lmu_theta = np.arctan((self.p_rest*sintheta)/(self.beta_gamma*self.e_rest + self.gamma*self.p_rest*costheta))
        lmu_energy = self.gamma*self.e_rest + self.beta_gamma*self.p_rest*costheta
        
        return lmu_energy, lmu_theta, phi, costheta, sintheta
    
    def neutrino_et(self, mu_energy, costheta, sintheta, mu_phi):
        """
        Calculates the lab energy and angle of the neutrino
        
        Parameters
        ----------
        mu_energy : float
            lab frame muon energy
        costheta : float
            cos of muon angle
        sintheta : float
            sin of muon angle
        mu_phi : float
            rotational angle of muon
        
        Returns
        -------
        ln_energy : float
            lab frame energy of the neutrino
        ln_theta : float
            lab frame angle of the neutrino
        phi : float
            rotational angle of the neutrino.

        """
        ln_energy = self.gamma*self.p_rest - self.beta_gamma*self.p_rest*costheta
        ln_theta = np.arctan((self.p_rest*sintheta)/(self.beta_gamma*self.p_rest-self.gamma*self.p_rest*costheta))
        phi = -mu_phi
        
        return ln_energy, ln_theta, phi
        
    def test_detector(self, detector, part):
        """
        Tests whether a hypothetical particle goes through a specified detector's geometry.
        
        Parameters
        ----------
        detector : class object
            detector object.
        part : class object
            hypothetical particle.

        Returns
        -------
        bool
            returns true if the particle goes through detector geometry.
        """
        i = 0
        for r in detector.range:
            
            initial = r[0]
            final = r[1]
  
            posi = part.move(initial)
            posf = part.move(final)
            
            t1 = detector.geometry[i].fit(posi)
            t2 = detector.geometry[i].fit(posf)
            
            i+=1
            
            if not t1 and not t2:
                return False
        
        return True
    
    class lav:
        #decays in this region. Adding for the sake atm but kinda useless
        def __init__(self):
            self.range = [[102.4,183.316]]
            self.geometry = []
            return None
    
    class straw:
        def __init__(self):
            self.range = [[183.316,218.983]]
            
            ginfo = {
                'range':self.range[0],
                'pos': [0,0,0],
                'radius': 1.05,
                'inner_radius': .06
                }
            
            self.geometry = [geometry.circular(ginfo)]
            return None
        
    class rich:
        def __init__(self):
            self.range = [[220.252, 223.73],[223.64,227.34],[227.43,232.35],[232.35,237.252]]
            
            ginfo0 = {
                'range':self.range[0],
                'pos': [0,0,0],
                'radius':1.9,
                'inner_radius': .084
                }
            
            ginfo1 = {
                'range':self.range[1],
                'pos': [0,0,0],
                'radius':1.81,
                'inner_radius': .084
                }
            
            ginfo2 = {
                'range':self.range[2],
                'pos': [0,0,0],
                'radius':1.676,
                'inner_radius': .084
                }
            ginfo3 = {
                'range':self.range[3],
                'pos': [0,0,0],
                'radius':1.625,
                'inner_radius': .084
                }
            
            self.geometry = [geometry.circular(ginfo0), geometry.circular(ginfo1), 
                             geometry.circular(ginfo2), geometry.circular(ginfo3)]
            return None
        
    class chod:
        def __init__(self):
            self.range = [[239.009,239.029],[239.389,239.409]]
            
            ginfo0 = {
                'range':self.range[0],
                'pos': [0,0,0],
                'radius':1.07,
                'inner_radius': .14
                }
            
            ginfo1 = {
                'range':self.range[1],
                'pos': [0,0,0],
                'radius':1.07,
                'inner_radius': .14
                }
            
            self.geometry = [geometry.circular(ginfo0), geometry.circular(ginfo1)]
            return None

    class muv3:
        def __init__(self):
            self.range = [[246.8, 246.85]]
            
            ginfo = {
                'range':self.range[0],
                'pos': [0,0,0],
                'width':1.32,
                'height':1.32,
                'inner_radius': .084
                }
            
            self.geometry = [geometry.rectangular(ginfo)]
            return None
        
    class lkr:
        def __init__(self):
            self.range = [[241.093,242.503]]
            
            ginfo = {
                'range':self.range[0],
                'pos': [0,0,0],
                'radius':1.28,
                'inner_radius': .04
                }
            
            self.geometry = [geometry.circular(ginfo)]
            return None

class analyse:
    def __init__(self, data):
        """
        Analyses the data given and allows for the creation of graphs

        Parameters
        ----------
        data : dictionary
            Data to be analysed.
        """
        m_energies = [me.get('m_energy') for me in data]
        n_energies = [ne.get('n_energy') for ne in data]
        m_theta = [mt.get('m_theta') for mt in data]
        n_theta = [nt.get('n_theta') for nt in data]
        origin = [mo.get('origin') for mo in data]
        
        self.distributions = {'Muon Energy' : m_energies,
                              'Neutrino Energy' : n_energies,
                              'Muon Angle' : m_theta,
                              'Neutrino Angle' : n_theta,
                              'Origin': origin,
                              }
        
    def distribution(self, data):
        """
        Creates a graph of the data with a given distribution
        
        Parameters
        ----------
        data : array
            data = [
            info = {
                    distribution_type : string (ie. histogram)
                    dists : [distx, disty] (self.distrubtion)
                    units : [unitx, unity] (units of stuff)
                    bins : int (used if bins)
                    }
            ].
        """
        for info in data:
            if info['distribution_type'] == 'histogram':    
                x = self.distributions[f'{info["dists"][0]}']
            
                counts, bins = np.histogram(x, bins = info['bins']) 
                plt.stairs(counts, bins)
            
                plt.xlabel(f'{info["dists"][0]} ({info["units"][0]})')
                plt.ylabel('Count')
                plt.title(f'Distribution of Accepted {info["dists"][0]}')
            
                plt.show()
            
            elif info['distribution_type'] == 'scatter':
                x = self.distributions[f'{info["dists"][0]}']
                y = self.distributions[f'{info["dists"][1]}']
            
                fig, ax = plt.subplots()
                
                ax.scatter(x,y, s = .1)
                
                plt.xlabel(f'{info["dists"][0]} ({info["units"][0]})')
                plt.ylabel(f'{info["dists"][1]} ({info["units"][1]})')
                plt.title(f'{info["dists"][0]} vs {info["dists"][1]}')
                
                plt.show()
        return
    
detector = na62()

nevents = 100000 #roughly 3 seconds for 50,000 events
nsuccess, et_data = detector.simulate(nevents)
print(nsuccess/nevents)


analyse(et_data[0]).distribution([
                        {'distribution_type' : 'histogram',
                        'dists' : ['Muon Angle'],
                        'units' : ['Radians'],
                        'bins' : 100,
                        },
      
                        {'distribution_type' : 'histogram',
                         'dists' : ['Muon Energy'],
                         'units' : ['GeV'],
                         'bins' : 100,
                         },
                        
                        {'distribution_type' : 'histogram',
                         'dists' : ['Neutrino Energy'],
                         'units' : ['GeV'],
                         'bins' : 100,
                         },
                        
                        {'distribution_type' : 'histogram',
                         'dists' : ['Neutrino Angle'],
                         'units' : ['Radians'],
                         'bins' : 100,
                         },
                        
                        {'distribution_type' : 'histogram',
                         'dists' : ['Origin'],
                         'units' : ['Meters'],
                         'bins' : 100,
                         },
                        
                        {'distribution_type' : 'scatter',
                         'dists' : ['Neutrino Angle','Neutrino Energy'],
                         'units' : ['Radians', 'GeV'],
                         'bins' : 100,
                         },
                        
                        ])
    