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
            return cv_state

        if engine_state.current_chamber > 0:
            cv_state.current_chamber = engine_state.current_chamber

        # Gun State fusion:
        # If CV detected DUAL_PORTAL from real HUD reticle brackets, keep DUAL_PORTAL!
        if cv_state.player.gun_state == PortalGunState.DUAL_PORTAL:
            pass
        elif engine_state.gun_state is not None:
            cv_state.player.gun_state = engine_state.gun_state
        
        # If player has NO gun
        if cv_state.player.gun_state == PortalGunState.NO_GUN:
            cv_state.portals.blue_active = False
            cv_state.portals.orange_active = False
            cv_state.player.crosshair.blue_ring_filled = False
            cv_state.player.crosshair.orange_ring_filled = False

        # Button & Door ground-truth overrides from engine
        cv_state.button_pressed = engine_state.button_pressed
        cv_state.exit_door_open = engine_state.door_open
        if engine_state.holding_cube:
            cv_state.player.holding_cube = True
        
        # Portal state fusion
        if cv_state.player.gun_state != PortalGunState.NO_GUN:
            if engine_state.blue_portal_placed:
                cv_state.portals.blue_active = True
                cv_state.player.crosshair.blue_ring_filled = True

            if engine_state.orange_portal_placed and cv_state.player.gun_state == PortalGunState.DUAL_PORTAL:
                cv_state.portals.orange_active = True
                cv_state.player.crosshair.orange_ring_filled = True

        if engine_state.player_dead:
            cv_state.death_detected = True

        if engine_state.level_complete:
            cv_state.level_complete = True

        return cv_state
