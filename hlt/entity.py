import abc

from . import commands, constants
from .positionals import Direction, Position

class Entity(abc.ABC):
    """
    Base Entity Class from whence Ships, Dropoffs and Shipyards inherit
    """
    def __init__(self, owner, id, position):
        self.owner = owner
        self.id = id
        self.position = position

    @staticmethod
    def _generate(player_id):
        """
        Method which creates an entity for a specific player given input from the engine.
        :param player_id: The player id for the player who owns this entity
        :return: An instance of Entity along with its id
        """
        ship_id, x_position, y_position = map(int, input().split())
        return ship_id, Entity(player_id, ship_id, Position(x_position, y_position))

    def __repr__(self):
        return "{}(id={}, {})".format(self.__class__.__name__,
                                      self.id,
                                      self.position)


class Dropoff(Entity):
    """
    Dropoff class for housing dropoffs
    """
    pass


class Shipyard(Entity):
    """
    Shipyard class to house shipyards
    """
    def spawn(self):
        """Return a move to spawn a new ship."""
        return commands.GENERATE


class Ship(Entity):
    """
    Ship class to house ship entities
    """
    def __init__(self, owner, id, position, halite_amount):
        super().__init__(owner, id, position)
        self.halite_amount = halite_amount

    @property
    def is_full(self):
        """Is this ship at max halite capacity?"""
        return self.halite_amount >= constants.MAX_HALITE

    def make_dropoff(self):
        """Return a move to transform this ship into a dropoff."""
        return "{} {}".format(commands.CONSTRUCT, self.id)

    def move(self, direction):
        """
        Return a move to move this ship in a direction without
        checking for collisions.
        """
        raw_direction = direction
        if not isinstance(direction, str) or direction not in "nsew":
            raw_direction = Direction.convert(direction)
        return "{} {} {}".format(commands.MOVE, self.id, raw_direction)

    def stay_still(self):
        """
        Don't move this ship.
        """
        return "{} {} {}".format(commands.MOVE, self.id, commands.STAY_STILL)

    @staticmethod
    def _generate(player_id):
        """
        Creates an instance of a ship for a given player given the engine's input.
        :param player_id: The id of the player who owns this ship
        :return: The ship id and ship object
        """
        ship_id, x_position, y_position, halite = map(int, input().split())
        return ship_id, Ship(player_id, ship_id, Position(x_position, y_position), halite)

    def __repr__(self):
        return "{}(id={}, {}, cargo={} halite)".format(self.__class__.__name__,
                                                       self.id,
                                                       self.position,
                                                       self.halite_amount)
