"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Simulation class
----------------

"""
from abc import abstractmethod

class Simulation(object):

    def __init__(self, aircraft, system, atmosphere, gravity):
        self.aircraft = aircraft
        self.system = system
        self.atmosphere = atmosphere
        self.gravity = gravity
        self._time_step = 0

    def time_step(self, dt):

        self._time_step += 1
        self.system.propagate(self.aircraft, dt)
        self.atmosphere(self.system)
        self.system.set_atmosphere()  # TODO: set API
        gravity_accel = self.gravity.get_gravity(self.system)
        gravity_force = gravity_accel * self.aircraft.mass
        controls = self.get_current_aircraft_controls()
        self.aircraft.set_controls(controls)
        forces, moments = self.aircraft.get_forces_and_moments(self.system)
        forces += gravity_force
        self.system.set_forces_and_moments(forces, moments)

    @abstractmethod
    def get_current_aircraft_controls(self, ii):
        return


class BatchSimulation(Simulation):

    def __init__(self, aircraft, system, atmosphere, gravity):
        Simulation.__init__(aircraft, system, atmosphere, gravity)
        self.time = None
        self.aircraft_controls = {}

    def set_aircraft_controls(self, time, controls):
        """Set the time history of controls and the corresponding times.

        Parameters
        ----------
        time
        controls

        Returns
        -------

        """

        # check time dimensions
        if time.ndim != 1:
            raise ValueError('Time must be unidimensional')
        tsize = time.size
        # check dimensions
        for c in controls:
            if c.size != tsize:
                msg = 'Control {} size ({}) does not match time size ({' \
                      '})'.fromat(c, c.size, tsize)
                raise ValueError(msg)
        self.time = time
        self.aircraft_controls = controls

    def get_current_aircraft_controls(self, ii):
        """Returns controls at time step ii.

        Parameters
        ----------
        ii

        Returns
        -------

        """
        return {control: self.aircraft_controls[control][ii] for control in \
                self.aircraft_controls.keys()}

    def run_simulation(self):

        for ii, t in enumerate(self.time[1:]):
            dt = t - self.time[ii-1]
            self.time_step(dt)



class RealTimeSimulation(Simulation):

    def __init__(self, aircraft, system, atmosphere, gravity):
        super(RealTimeSimulation, self).__init__(aircraft, system, atmosphere,
                                                 gravity)
        # TODO:...

    def get_current_aircraft_controls(self, ii):
        # Joystick reading
        raise NotImplementedError
