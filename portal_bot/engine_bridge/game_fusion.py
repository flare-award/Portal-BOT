"""State fusion combining Computer Vision with Game Engine Console logs."""

from portal_bot.core.types import GameState, PortalGunState, PortalState
from portal_bot.engine_bridge.console_reader import ConsoleLogReader, EngineState


class StateFusion:
    """Combines CV perception with Ground Truth Engine logs."""

    def __init__(self, console_reader: ConsoleLogReader):
        self.console_reader = console_reader

    def fuse_states(self, cv_state: GameState) -> GameState:
        engine_state: EngineState = self.console_reader.state

        if not engine_state.connected:
            # Fall back purely to Computer Vision if engine log is not hooked
            return cv_state

        # 1. Gun State override from Engine/Map
        cv_state.player.gun_state = engine_state.gun_state
        cv_state.current_chamber = engine_state.current_chamber

        # 2. Portal state fusion:
        # If player has NO gun, portals cannot be fired by player!
        if engine_state.gun_state == PortalGunState.NO_GUN:
            cv_state.portals.blue_active = False
            cv_state.portals.orange_active = False
            cv_state.player.crosshair.blue_ring_filled = False
            cv_state.player.crosshair.orange_ring_filled = False

        elif engine_state.gun_state == PortalGunState.SINGLE_PORTAL_BLUE:
            # Orange portal is static / fixed in chamber
            cv_state.portals.orange_active = True
            cv_state.portals.blue_active = engine_state.blue_portal_placed
            cv_state.player.crosshair.orange_ring_filled = True
            cv_state.player.crosshair.blue_ring_filled = engine_state.blue_portal_placed

        elif engine_state.gun_state == PortalGunState.DUAL_PORTAL:
            cv_state.portals.blue_active = engine_state.blue_portal_placed
            cv_state.portals.orange_active = engine_state.orange_portal_placed
            cv_state.player.crosshair.blue_ring_filled = engine_state.blue_portal_placed
            cv_state.player.crosshair.orange_ring_filled = engine_state.orange_portal_placed

        # 3. Floor Button & Door state override
        cv_state.button_pressed = engine_state.button_pressed
        cv_state.exit_door_open = engine_state.door_open
        
        # 4. Death & Level Completion
        if engine_state.player_dead:
            cv_state.death_detected = True

        if engine_state.level_complete:
            cv_state.level_complete = True

        return cv_state
