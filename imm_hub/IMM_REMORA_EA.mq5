//+------------------------------------------------------------------+
//|  IMM_REMORA_EA.mq5                                               |
//|  APEX REMORA — Phase III Executor                                |
//|  Connects to IMM AGI Hub via ngrok WebSocket.                    |
//|  Executes Phase III revival signals with dynamic exit mgmt.      |
//|                                                                  |
//|  Author: Travis Bergen | IMM Framework                           |
//|  Zenodo: doi.org/10.5281/zenodo.19075097                         |
//+------------------------------------------------------------------+
#property copyright "Travis Bergen"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//-- Inputs
input string HubURL        = "ws://YOUR_NGROK_URL/ea/remora";  // Hub WebSocket URL
input string EaId          = "remora";
input int    HeartbeatSecs = 10;
input double MinLots       = 0.01;
input double MaxLots       = 10.0;
input int    MaxSlippage   = 5;     // pips
input bool   VerboseLog    = true;

//-- State
CTrade   g_trade;
string   g_trade_id   = "";
double   g_entry_price = 0;
int      g_direction   = 0;   // 1=BUY -1=SELL
double   g_lots        = 0;
double   g_current_stop = 0;
bool     g_in_trade    = false;
datetime g_last_hb     = 0;

// WebSocket handle (MT5 built-in - available in recent builds)
// For older builds use WinInet or a DLL bridge
int g_ws_handle = INVALID_HANDLE;

//+------------------------------------------------------------------+
int OnInit() {
   g_trade.SetExpertMagicNumber(202407);
   g_trade.SetDeviationInPoints(MaxSlippage * 10);

   g_ws_handle = WebSocketCreate(HubURL);
   if (g_ws_handle == INVALID_HANDLE) {
      Print("[REMORA] WebSocket connection failed to: ", HubURL);
      Print("[REMORA] Retrying in OnTick...");
   } else {
      Print("[REMORA] Connected to hub: ", HubURL);
      SendHeartbeat();
   }
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   if (g_ws_handle != INVALID_HANDLE)
      WebSocketClose(g_ws_handle);
}

//+------------------------------------------------------------------+
void OnTick() {
   // Reconnect if dropped
   if (g_ws_handle == INVALID_HANDLE) {
      g_ws_handle = WebSocketCreate(HubURL);
      if (g_ws_handle != INVALID_HANDLE) {
         Print("[REMORA] Reconnected.");
         SendHeartbeat();
      }
      return;
   }

   // Send tick state to hub
   SendTickUpdate();

   // Heartbeat
   if (TimeCurrent() - g_last_hb >= HeartbeatSecs) {
      SendHeartbeat();
      g_last_hb = TimeCurrent();
   }

   // Read commands from hub
   string msg = "";
   while (WebSocketRead(g_ws_handle, msg)) {
      if (msg != "") ProcessCommand(msg);
   }
}

//+------------------------------------------------------------------+
void SendTickUpdate() {
   string json = StringFormat(
      "{\"type\":\"tick_update\","
      "\"symbol\":\"%s\","
      "\"bid\":%.5f,"
      "\"ask\":%.5f,"
      "\"time\":%d,"
      "\"trade_id\":\"%s\","
      "\"open_pnl\":%.2f}",
      Symbol(),
      SymbolInfoDouble(Symbol(), SYMBOL_BID),
      SymbolInfoDouble(Symbol(), SYMBOL_ASK),
      (int)TimeCurrent(),
      g_trade_id,
      AccountInfoDouble(ACCOUNT_PROFIT)
   );
   WebSocketSend(g_ws_handle, json);
}

//+------------------------------------------------------------------+
void SendHeartbeat() {
   string json = StringFormat(
      "{\"type\":\"heartbeat\",\"ea_id\":\"%s\",\"time\":%d}",
      EaId, (int)TimeCurrent()
   );
   WebSocketSend(g_ws_handle, json);
}

//+------------------------------------------------------------------+
void ProcessCommand(string json) {
   if (VerboseLog) Print("[REMORA] CMD: ", json);

   // Simple JSON key extraction (no external library needed)
   string cmd_type = ExtractString(json, "\"type\"");

   if (cmd_type == "ENTER")          HandleEnter(json);
   else if (cmd_type == "CLOSE_ALL") HandleCloseAll(ExtractString(json, "\"reason\""));
   else if (cmd_type == "CLOSE_PARTIAL") HandleClosePartial(ExtractDouble(json, "\"fraction\""));
   else if (cmd_type == "TRAIL_STOP")    HandleTrailStop(ExtractDouble(json, "\"new_stop\""));
   else if (cmd_type == "MOVE_TO_BE_PLUS") HandleMoveToBePlus(ExtractDouble(json, "\"new_stop\""));
}

//+------------------------------------------------------------------+
void HandleEnter(string json) {
   if (g_in_trade) {
      Print("[REMORA] Already in trade — ignoring ENTER");
      return;
   }

   string symbol    = ExtractString(json, "\"symbol\"");
   string dir_str   = ExtractString(json, "\"direction\"");
   double lots      = NormaliseLots(ExtractDouble(json, "\"lots\""));
   int    sl_pips   = (int)ExtractDouble(json, "\"initial_sl_pips\"");
   g_trade_id       = ExtractString(json, "\"trade_id\"");

   if (symbol != Symbol()) {
      Print("[REMORA] Symbol mismatch: got ", symbol, " expected ", Symbol());
      return;
   }

   int order_type   = (dir_str == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   double ask       = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid       = SymbolInfoDouble(symbol, SYMBOL_BID);
   double entry     = (order_type == ORDER_TYPE_BUY) ? ask : bid;
   double point     = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double sl_price  = (order_type == ORDER_TYPE_BUY)
                        ? entry - sl_pips * point * 10
                        : entry + sl_pips * point * 10;

   bool ok = (order_type == ORDER_TYPE_BUY)
      ? g_trade.Buy(lots, symbol, 0, sl_price, 0, "IMM_REMORA")
      : g_trade.Sell(lots, symbol, 0, sl_price, 0, "IMM_REMORA");

   if (ok) {
      g_in_trade      = true;
      g_entry_price   = entry;
      g_direction     = (order_type == ORDER_TYPE_BUY) ? 1 : -1;
      g_lots          = lots;
      g_current_stop  = sl_price;
      Print("[REMORA] ENTERED: ", dir_str, " ", lots, " lots @ ", entry,
            " SL=", sl_price, " ID=", g_trade_id);

      // Confirm to hub
      string confirm = StringFormat(
         "{\"type\":\"trade_opened\",\"trade_id\":\"%s\","
         "\"symbol\":\"%s\",\"entry_price\":%.5f,\"lots\":%.2f}",
         g_trade_id, symbol, entry, lots
      );
      WebSocketSend(g_ws_handle, confirm);
   } else {
      Print("[REMORA] Order failed: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
void HandleCloseAll(string reason) {
   if (!g_in_trade) return;
   double close_price = (g_direction == 1)
      ? SymbolInfoDouble(Symbol(), SYMBOL_BID)
      : SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   ClosePosition(g_lots, close_price);
   Print("[REMORA] CLOSED ALL — reason: ", reason);
}

//+------------------------------------------------------------------+
void HandleClosePartial(double fraction) {
   if (!g_in_trade || fraction <= 0) return;
   double close_lots = NormaliseLots(g_lots * fraction);
   if (close_lots < MinLots) close_lots = g_lots;  // close all if remainder tiny

   double close_price = (g_direction == 1)
      ? SymbolInfoDouble(Symbol(), SYMBOL_BID)
      : SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   ClosePosition(close_lots, close_price);
   g_lots -= close_lots;
   if (g_lots < MinLots) g_in_trade = false;
}

//+------------------------------------------------------------------+
void HandleTrailStop(double new_stop) {
   if (!g_in_trade || new_stop <= 0) return;

   // Only move stop in profit direction
   bool should_update = (g_direction == 1)
      ? (new_stop > g_current_stop)   // BUY: only move up
      : (new_stop < g_current_stop);  // SELL: only move down

   if (!should_update) return;

   // Modify the open position stop
   ulong ticket = GetOpenTicket();
   if (ticket > 0) {
      g_trade.PositionModify(ticket, new_stop, 0);
      g_current_stop = new_stop;
      if (VerboseLog) Print("[REMORA] TRAIL → SL=", new_stop);
   }
}

//+------------------------------------------------------------------+
void HandleMoveToBePlus(double be_stop) {
   if (!g_in_trade || be_stop <= 0) return;
   ulong ticket = GetOpenTicket();
   if (ticket > 0) {
      g_trade.PositionModify(ticket, be_stop, 0);
      g_current_stop = be_stop;
      Print("[REMORA] MOVED TO BE+ SL=", be_stop);
   }
}

//+------------------------------------------------------------------+
void ClosePosition(double lots_to_close, double close_price) {
   ulong ticket = GetOpenTicket();
   if (ticket == 0) { g_in_trade = false; return; }

   bool ok = (g_direction == 1)
      ? g_trade.Sell(lots_to_close, Symbol(), 0, 0, 0, "IMM_CLOSE")
      : g_trade.Buy(lots_to_close,  Symbol(), 0, 0, 0, "IMM_CLOSE");

   if (ok) {
      string confirm = StringFormat(
         "{\"type\":\"trade_closed\",\"trade_id\":\"%s\","
         "\"close_price\":%.5f,\"lots\":%.2f}",
         g_trade_id, close_price, lots_to_close
      );
      WebSocketSend(g_ws_handle, confirm);
      if (lots_to_close >= g_lots) {
         g_in_trade = false;
         g_trade_id = "";
      }
   }
}

//+------------------------------------------------------------------+
ulong GetOpenTicket() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if (PositionSelectByTicket(ticket)) {
         if (PositionGetString(POSITION_SYMBOL) == Symbol() &&
             PositionGetInteger(POSITION_MAGIC) == 202407)
            return ticket;
      }
   }
   return 0;
}

//+------------------------------------------------------------------+
double NormaliseLots(double lots) {
   double step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   lots = MathRound(lots / step) * step;
   return MathMax(MinLots, MathMin(MaxLots, lots));
}

//-- Minimal JSON helpers (no external library dependency)
string ExtractString(string json, string key) {
   int start = StringFind(json, key);
   if (start < 0) return "";
   start = StringFind(json, "\"", start + StringLen(key) + 1);
   if (start < 0) return "";
   int end = StringFind(json, "\"", start + 1);
   if (end < 0) return "";
   return StringSubstr(json, start + 1, end - start - 1);
}

double ExtractDouble(string json, string key) {
   int start = StringFind(json, key);
   if (start < 0) return 0;
   start += StringLen(key) + 1;
   while (start < StringLen(json) && (StringGetCharacter(json, start) == ' ' ||
          StringGetCharacter(json, start) == ':')) start++;
   string num = "";
   while (start < StringLen(json)) {
      ushort c = StringGetCharacter(json, start);
      if (c == ',' || c == '}' || c == ' ') break;
      num += ShortToString(c);
      start++;
   }
   return StringToDouble(num);
}
