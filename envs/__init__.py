import gymnasium as gym


def register():
    gym.register(
        id="TerrainAnt-v0",
        entry_point="envs.terrain_ant:TerrainAntEnv",
        max_episode_steps=1000,
    )
    gym.register(
        id="TerrainAnt-v1",
        entry_point="envs.terrain_ant:VelocityTerrainAntEnv",
        max_episode_steps=1000,
    )
    gym.register(
        id="DamageAnt-v0",
        entry_point="envs.damage_ant:DamageAntEnv",
        max_episode_steps=1000,
    )
