#!/usr/bin/env python

import os
import sys

# Resolve path configuration
root = os.path.join(os.path.dirname(__file__), "..")
src = os.path.join(root, "pypokergui")
sys.path.append(root)
sys.path.append(src)

import click
import webbrowser

from pypokergui.server.poker import start_server
from pypokergui.config_builder import build_config

@click.group()
def cli():
    pass

@cli.command(name="serve")
@click.argument("config")
@click.option("--port", default=8000, help="port to run server")
@click.option("--speed", default="moderate", type=click.Choice(["slow", "moderate", "fast"]), help="how fast game progress")
def serve_command(config, port, speed):
    host = "localhost"
    webbrowser.open("http://%s:%s" % (host, port))
    start_server(config, port, speed)

@cli.command(name="build_config")
@click.option("-p", "--players", default=4, help="number of players")
@click.option("-r", "--maxround", default=10, help="final round of the game")
@click.option("-s", "--stack", default=100, help="start stack of player")
@click.option("-b", "--small_blind", default=5, help="amount of small blind")
@click.option("-a", "--ante", default=0, help="amount of ante")
@click.option("-m", "--model", help="path of the model, either absolute or relative to directory")
def build_config_command(players, maxround, stack, small_blind, ante, model):
    build_config(players, maxround, stack, small_blind, ante, model)

if __name__ == '__main__':
    cli()
