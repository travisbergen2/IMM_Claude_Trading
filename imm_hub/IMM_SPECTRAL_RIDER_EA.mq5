//+------------------------------------------------------------------+
//|  IMM_SPECTRAL_RIDER_EA.mq5                                       |
//|  SPECTRAL RIDER — Phase II Monitor / Pre-Position                |
//|  Watches Phase II compression and enters probe positions.        |
//|  Connects to IMM AGI Hub via ngrok WebSocket.                    |
//|                                                                  |
//|  Author: Travis Bergen | IMM Framework                           |
//|  Zenodo: doi.org/10.5281/zenodo.19075097                         |
//+------------------------------------------------------------------+
#property copyright "Travis Bergen"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input string HubURL        = "ws://YOUR_NGROK_URL/ea/spectral_rider";
input string EaId          = "spectral_rider";
input int    HeartbeatSecs = 10;
input double MinLots       = 0.01;
input bool   VerboseLog    = true;

CTrade   g_trade;
string   g_trade_id    = "";
double   g_entry_price = 0;
int      g_direction   = 0;
double   g_lots        = 0;
double   g_current_stop = 0;
bool     g_in_trade    = false;
datetime g_last_hb     = 0;
int      g_ws_handle   = INVALID_HANDLE;

int OnInit() {
   g_trade.SetExpertMagicNumber(202408);
   g_trade.SetDeviationInPoints(30);

   g_ws_handle = WebSocketCreate(HubURL);
   if (g_ws_handle == INVALID_HANDLE)
      Print("[SPECTRAL] Hub connection failed — will retry.");
   else
      Print("[SPECTRAL] Connected: ", HubURL);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if (g_ws_handle != INVALID_HANDLE)
      WebSocketClose(g_ws_handle);
}

void OnTick() {
   if (g_ws_handle == INVALID_HANDLE) {
      g_ws_handle = WebSocketCreate(HubURL);
      return;
   }

   SendTickUpdate();

   if (TimeCurrent() - g_last_hb >= HeartbeatSecs) {
      SendHeartbeat();
      g_last_hb = TimeCurrent();
   }

   string msg = "";
   while (WebSocketRead(g_ws_handle, msg)) {
      if (msg != "") ProcessCommand(msg);
   }
}

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

void SendHeartbeat() {
   string json = StringFormat(
      "{\"type\":\"heartbeat\",\"ea_id\":\"%s\",\"time\":%d}",
      EaId, (int)TimeCurrent()
   );
   WebSocketSend(g_ws_handle, json);
}

void ProcessCommand(string json) {
   if (VerboseLog) Print("[SPECTRAL] CMD: ", json);
   string cmd_type = ExtractString(json, "\"type\"");

   if (cmd_type == "ENTER")              HandleProbeEnter(json);
   else if (cmd_type == "CLOSE_ALL")     HandleCloseAll();
   else if (cmd_type == "CLOSE_PARTIAL") HandleClosePartial(ExtractDouble(json, "\"fraction\""));
   else if (cmd_type == "TRAIL_STOP")    HandleTrailStop(ExtractDouble(json, "\"new_stop\""));
   else if (cmd_type == "MOVE_TO_BE_PLUS") HandleMoveToBePlus(ExtractDouble(json, "\"new_stop\""));
}

void HandleProbeEnter(string json) {
   if (g_in_trade) return;

   string symbol  = ExtractString(json, "\"symbol\"");
   string dir_str = ExtractString(json, "\"direction\"");
   double lots    = NormaliseLots(ExtractDouble(json, "\"lots\""));
   int    sl_pips = (int)ExtractDouble(json, "\"initial_sl_pips\"");
   g_trade_id     = ExtractString(json, "\"trade_id\"");

   if (symbol != Symbol()) return;

   // Probe enters on limit if offset provided, else market
   double offset_pips = ExtractDouble(json, "\"limit_offset_pips\"");

   int order_type = (dir_str == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double entry = (order_type == ORDER_TYPE_BUY) ? ask : bid;

   double sl_price = (order_type == ORDER_TYPE_BUY)
      ? entry - sl_pips * point * 10
      : entry + sl_pips * point * 10;

   bool ok;
   if (offset_pips > 0) {
      // Limit order slightly in the direction of move
      double limit_price = (order_type == ORDER_TYPE_BUY)
         ? ask - offset_pips * point * 10
         : bid + offset_pips * point * 10;
      ok = (order_type == ORDER_TYPE_BUY)
         ? g_trade.BuyLimit(lots, limit_price, symbol, sl_price, 0, 0, 0, "IMM_PROBE")
         : g_trade.SellLimit(lots, limit_price, symbol, sl_price, 0, 0, 0, "IMM_PROBE");
   } else {
      ok = (order_type == ORDER_TYPE_BUY)
         ? g_trade.Buy(lots, symbol, 0, sl_price, 0, "IMM_PROBE")
         : g_trade.Sell(lots, symbol, 0, sl_price, 0, "IMM_PROBE");
   }

   if (ok) {
      g_in_trade     = true;
      g_entry_price  = entry;
      g_direction    = (order_type == ORDER_TYPE_BUY) ? 1 : -1;
      g_lots         = lots;
      g_current_stop = sl_price;
      Print("[SPECTRAL] PROBE ENTERED: ", dir_str, " ", lots, " @ ", entry);
   }
}

void HandleCloseAll() {
   if (!g_in_trade) return;
   ClosePosition(g_lots);
}

void HandleClosePartial(double fraction) {
   if (!g_in_trade || fraction <= 0) return;
   double close_lots = NormaliseLots(g_lots * fraction);
   if (close_lots >= g_lots) { ClosePosition(g_lots); return; }
   ClosePosition(close_lots);
   g_lots -= close_lots;
   if (g_lots < MinLots) g_in_trade = false;
}

void HandleTrailStop(double new_stop) {
   if (!g_in_trade) return;
   bool ok = (g_direction == 1) ? (new_stop > g_current_stop) : (new_stop < g_current_stop);
   if (!ok) return;
   ulong ticket = GetOpenTicket();
   if (ticket > 0) { g_trade.PositionModify(ticket, new_stop, 0); g_current_stop = new_stop; }
}

void HandleMoveToBePlus(double be_stop) {
   if (!g_in_trade) return;
   ulong ticket = GetOpenTicket();
   if (ticket > 0) { g_trade.PositionModify(ticket, be_stop, 0); g_current_stop = be_stop; }
}

void ClosePosition(double lots) {
   ulong ticket = GetOpenTicket();
   if (ticket == 0) { g_in_trade = false; return; }
   bool ok = (g_direction == 1)
      ? g_trade.Sell(lots, Symbol(), 0, 0, 0, "IMM_CLOSE")
      : g_trade.Buy(lots, Symbol(), 0, 0, 0, "IMM_CLOSE");
   if (ok && lots >= g_lots) { g_in_trade = false; g_trade_id = ""; }
}

ulong GetOpenTicket() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong t = PositionGetTicket(i);
      if (PositionSelectByTicket(t) &&
          PositionGetString(POSITION_SYMBOL) == Symbol() &&
          PositionGetInteger(POSITION_MAGIC) == 202408) return t;
   }
   return 0;
}

double NormaliseLots(double lots) {
   double step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   lots = MathRound(lots / step) * step;
   return MathMax(MinLots, MathMin(5.0, lots));
}

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
      num += ShortToString(c); start++;
   }
   return StringToDouble(num);
}
