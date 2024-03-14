from nle import nethack
from nle.nethack import actions as nethack_actions
from jax import Array


def decode_observation(obs: Array, separator: str=" ") -> str:
    rows, cols = obs.shape
    result = ""
    for i in range(rows):
        result += "\n" + separator.join(map(chr, obs[i]))
    return result


def decode_message(message):
    if int(message.sum()) == 0:
        return ""
    return "".join(map(chr, message))


def decode_action(action: int):
    actions = [
        nethack.CompassCardinalDirection.N,  # 0
        nethack.CompassCardinalDirection.E,  # 1
        nethack.CompassCardinalDirection.S,  # 2
        nethack.CompassCardinalDirection.W,  # 3
        nethack.Command.PICKUP,  # 4
        nethack.Command.APPLY,  # 5
    ]
    nle_action = actions[action]
    return ACTIONS_MAP[nle_action][0]


ACTIONS_MAP = {
    None: ["none", ""],
    nethack_actions.UnsafeActions.HELP: ["help", "?"],
    nethack_actions.UnsafeActions.PREVMSG: ["previous message", "^p"],
    nethack_actions.CompassDirection.N: ["north", "k"],  # type: ignore
    nethack_actions.CompassDirection.E: ["east", "l"],  # type: ignore
    nethack_actions.CompassDirection.S: ["south", "j"],  # type: ignore
    nethack_actions.CompassDirection.W: ["west", "h"],  # type: ignore
    nethack_actions.CompassIntercardinalDirection.NE: ["northeast", "u"],
    nethack_actions.CompassIntercardinalDirection.SE: ["southeast", "n"],
    nethack_actions.CompassIntercardinalDirection.SW: ["southwest", "b"],
    nethack_actions.CompassIntercardinalDirection.NW: ["northwest", "y"],
    nethack_actions.CompassCardinalDirectionLonger.N: ["far north", "K"],
    nethack_actions.CompassCardinalDirectionLonger.E: ["far east", "L"],
    nethack_actions.CompassCardinalDirectionLonger.S: ["far south", "J"],
    nethack_actions.CompassCardinalDirectionLonger.W: ["far west", "H"],
    nethack_actions.CompassIntercardinalDirectionLonger.NE: ["far northeast", "U"],
    nethack_actions.CompassIntercardinalDirectionLonger.SE: ["far southeast", "N"],
    nethack_actions.CompassIntercardinalDirectionLonger.SW: ["far southwest", "B"],
    nethack_actions.CompassIntercardinalDirectionLonger.NW: ["far northwest", "Y"],
    nethack_actions.MiscDirection.UP: ["up", "<"],
    nethack_actions.MiscDirection.DOWN: ["down", ">"],
    nethack_actions.MiscDirection.WAIT: ["wait", "."],
    nethack_actions.MiscAction.MORE: ["more", "\r", r"\r"],
    nethack_actions.Command.EXTCMD: ["extcmd", "#"],
    nethack_actions.Command.EXTLIST: ["extlist", "M-?"],
    nethack_actions.Command.ADJUST: ["adjust", "M-a"],
    nethack_actions.Command.ANNOTATE: ["annotate", "M-A"],
    nethack_actions.Command.APPLY: ["apply", "a"],
    nethack_actions.Command.ATTRIBUTES: ["attributes", "^x"],
    nethack_actions.Command.AUTOPICKUP: ["autopickup", "@"],
    nethack_actions.Command.CALL: ["call", "C"],
    nethack_actions.Command.CAST: ["cast", "Z"],
    nethack_actions.Command.CHAT: ["chat", "M-c"],
    nethack_actions.Command.CLOSE: ["close", "c"],
    nethack_actions.Command.CONDUCT: ["conduct", "M-C"],
    nethack_actions.Command.DIP: ["dip", "M-d"],
    nethack_actions.Command.DROP: ["drop", "d"],
    nethack_actions.Command.DROPTYPE: ["droptype", "D"],
    nethack_actions.Command.EAT: ["eat", "e"],
    nethack_actions.Command.ESC: ["esc", "^["],
    nethack_actions.Command.ENGRAVE: ["engrave", "E"],
    nethack_actions.Command.ENHANCE: ["enhance", "M-e"],
    nethack_actions.Command.FIRE: ["fire", "f"],
    nethack_actions.Command.FIGHT: ["fight", "F"],
    nethack_actions.Command.FORCE: ["force", "M-f"],
    nethack_actions.Command.GLANCE: ["glance", ";"],
    nethack_actions.Command.HISTORY: ["history", "V"],
    nethack_actions.Command.INVENTORY: ["inventory", "i"],
    nethack_actions.Command.INVENTTYPE: ["inventtype", "I"],
    nethack_actions.Command.INVOKE: ["invoke", "M-i"],
    nethack_actions.Command.JUMP: ["jump", "M-j"],
    nethack_actions.Command.KICK: ["kick", "^d"],
    nethack_actions.Command.KNOWN: ["known", "\\"],
    nethack_actions.Command.KNOWNCLASS: ["knownclass", "`"],
    nethack_actions.Command.LOOK: ["look", ":"],
    nethack_actions.Command.LOOT: ["loot", "M-l"],
    nethack_actions.Command.MONSTER: ["monster", "M-m"],
    nethack_actions.Command.MOVE: ["move", "m"],
    nethack_actions.Command.MOVEFAR: ["movefar", "M"],
    nethack_actions.Command.OFFER: ["offer", "M-o"],
    nethack_actions.Command.OPEN: ["open", "o"],
    nethack_actions.Command.OPTIONS: ["options", "O"],
    nethack_actions.Command.OVERVIEW: ["overview", "^o"],
    nethack_actions.Command.PAY: ["pay", "p"],
    nethack_actions.Command.PICKUP: ["pickup", ","],
    nethack_actions.Command.PRAY: ["pray", "M-p"],
    nethack_actions.Command.PUTON: ["puton", "P"],
    nethack_actions.Command.QUAFF: ["quaff", "q"],
    nethack_actions.Command.QUIT: ["quit", "M-q"],
    nethack_actions.Command.QUIVER: ["quiver", "Q"],
    nethack_actions.Command.READ: ["read", "r"],
    nethack_actions.Command.REDRAW: ["redraw", "^r"],
    nethack_actions.Command.REMOVE: ["remove", "R"],
    nethack_actions.Command.RIDE: ["ride", "M-R"],
    nethack_actions.Command.RUB: ["rub", "M-r"],
    nethack_actions.Command.RUSH: ["rush", "g"],
    nethack_actions.Command.RUSH2: ["rush2", "G"],
    nethack_actions.Command.SAVE: ["save", "S"],
    nethack_actions.Command.SEARCH: ["search", "s"],
    nethack_actions.Command.SEEALL: ["seeall", "*"],
    nethack_actions.Command.SEEAMULET: ["seeamulet", '"'],
    nethack_actions.Command.SEEARMOR: ["seearmor", "["],
    nethack_actions.Command.SEEGOLD: ["seegold", "dollar", "$"],
    nethack_actions.Command.SEERINGS: ["seerings", "="],
    nethack_actions.Command.SEESPELLS: ["seespells", "plus", "+"],
    nethack_actions.Command.SEETOOLS: ["seetools", "("],
    nethack_actions.Command.SEETRAP: ["seetrap", "^"],
    nethack_actions.Command.SEEWEAPON: ["seeweapon", ")"],
    nethack_actions.Command.SHELL: ["shell", "!"],
    nethack_actions.Command.SIT: ["sit", "M-s"],
    nethack_actions.Command.SWAP: ["swap", "x"],
    nethack_actions.Command.TAKEOFF: ["takeoff", "T"],
    nethack_actions.Command.TAKEOFFALL: ["takeoffall", "A"],
    nethack_actions.Command.TELEPORT: ["teleport", "^t"],
    nethack_actions.Command.THROW: ["throw", "t"],
    nethack_actions.Command.TIP: ["tip", "M-T"],
    nethack_actions.Command.TRAVEL: ["travel", "_"],
    nethack_actions.Command.TURN: ["turnundead", "M-t"],
    nethack_actions.Command.TWOWEAPON: ["twoweapon", "X"],
    nethack_actions.Command.UNTRAP: ["untrap", "M-u"],
    nethack_actions.Command.VERSION: ["version", "M-v"],
    nethack_actions.Command.VERSIONSHORT: ["versionshort", "v"],
    nethack_actions.Command.WEAR: ["wear", "W"],
    nethack_actions.Command.WHATDOES: ["whatdoes", "&"],
    nethack_actions.Command.WHATIS: ["whatis", "/"],
    nethack_actions.Command.WIELD: ["wield", "w"],
    nethack_actions.Command.WIPE: ["wipe", "M-w"],
    nethack_actions.Command.ZAP: ["zap", "z"],
    nethack_actions.TextCharacters.MINUS: ["minus", "-"],
    nethack_actions.TextCharacters.SPACE: ["space", " "],
    nethack_actions.TextCharacters.APOS: ["apos", "'"],
    nethack_actions.TextCharacters.NUM_0: ["zero", "0"],
    nethack_actions.TextCharacters.NUM_1: ["one", "1"],
    nethack_actions.TextCharacters.NUM_2: ["two", "2"],
    nethack_actions.TextCharacters.NUM_3: ["three", "3"],
    nethack_actions.TextCharacters.NUM_4: ["four", "4"],
    nethack_actions.TextCharacters.NUM_5: ["five", "5"],
    nethack_actions.TextCharacters.NUM_6: ["six", "6"],
    nethack_actions.TextCharacters.NUM_7: ["seven", "7"],
    nethack_actions.TextCharacters.NUM_8: ["eight", "8"],
    nethack_actions.TextCharacters.NUM_9: ["nine", "9"],
    nethack_actions.WizardCommand.WIZDETECT: ["wizard detect", "^e"],
    nethack_actions.WizardCommand.WIZGENESIS: ["wizard genesis", "^g"],
    nethack_actions.WizardCommand.WIZIDENTIFY: ["wizard identify", "^i"],
    nethack_actions.WizardCommand.WIZLEVELPORT: ["wizard teleport", "^v"],
    nethack_actions.WizardCommand.WIZMAP: ["wizard map", "^f"],
    nethack_actions.WizardCommand.WIZWHERE: ["wizard where", "^o"],
    nethack_actions.WizardCommand.WIZWISH: ["wizard wish", "^w"],
}
