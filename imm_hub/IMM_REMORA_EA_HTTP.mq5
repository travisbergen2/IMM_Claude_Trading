//+------------------------------------------------------------------+
//|  IMM_REMORA_EA_HTTP.mq5                                          |
//|  APEX REMORA — Phase III Executor (HTTP Polling Version)         |
//|  Uses WebRequest() — compatible with ALL MT5/TradeLocker builds  |
//|  on Windows. No WebSocket library required.                      |
//|                                                                  |
//|  SETUP:                                                          |
//|  1. Tools → Options → Expert Advisors                           |
//|  2. Check "Allow WebRequest for listed URL"                      |
//|  3. Add your ngrok URL (e.g. https://xxxx.ngrok-free.app)       |
//|  4. Set HubBaseURL input to same URL                             |
//|                                                                  |
//|  Author: Travis Bergen | IMM Framework                           |
//|  Zenodo: doi.org/10.5281/zenodo.19075097                         |
//+------------------------------------------------------------------+
#property copyright "Travis Bergen"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//── Inputs ──────────────────────────────────────────────────────────
input string   HubBaseURL     = "https://YOUR_NGROK_URL";  // Hub base URL (no trailing slash)
input string   EaId           = "remora";                   // EA identity key
input int      PollTimerMs    = 2000;    // Poll interval in milliseconds (2s)
input int      TickPostInterv = 3;       // Post tick every N ticks (reduce HTTP load)
input int      HttpTimeout    = 3000;    // WebRequest timeout ms
input double   MinLots        = 0.01;
input double   MaxLots        = 10.0;
input int      MaxSlippage    = 5;       // pips
input bool     VerboseLog     = false;

//── State ────────────────────────────────────────────────────────────
CTrade  g_trade;
string  g_trade_id    = "";
double  g_entry_price = 0.0;
int     g_direction   = 0;    // 1=BUY -1=SELL
double  g_lots        = 0.0;
double  g_current_stop = 0.0;
bool    g_in_trade    = false;
int     g_tick_count  = 0;

string  g_tick_url;
string  g_poll_url;
string  g_event_url;
string  g_hb_url;

//── Init ─────────────────────────────────────────────────────────────
int OnInit()
{
   g_trade.SetExpertMagicNumber(202501);
   g_trade.SetDeviationInPoints(MaxSlippage * 10);
   g_trade.SetTypeFillingBySymbol(Symbol());

   // Build URLs once
   g_tick_url  = HubBaseURL + "/ea/tick/"  + EaId;
   g_poll_url  = HubBaseURL + "/ea/poll/"  + EaId;
   g_event_url = HubBaseURL + "/ea/event/" + EaId;
   g_hb_url    = HubBaseURL + "/ea/heartbeat/" + EaId;

   // Timer drives polling (independent of tick frequency)
   EventSetMillisecondTimer(PollTimerMs);

   // Initial heartbeat
   SendHeartbeat();

   Print("[REMORA] Initialised. Hub: ", HubBaseURL);
   Print("[REMORA] Poll URL: ", g_poll_url);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("[REMORA] Deinitialised. Reason: ", reason);
}

//── OnTick: send market data ──────────────────────────────────────────
void OnTick()
{
   g_tick_count++;
   // Post tick every N ticks to avoid overwhelming the hub
   if(g_tick_count % TickPostInterv == 0)
      PostTick();
}

//── OnTimer: poll for commands ────────────────────────────────────────
void OnTimer()
{
   PollCommands();
}

//── HTTP helpers ──────────────────────────────────────────────────────

bool HttpPost(const string url, const string body, string &response)
{
   char   data[];
   char   result[];
   string headers;
   string req_headers = "Content-Type: application/json\r\n";

   StringToCharArray(body, data, 0, StringLen(body));
   ArrayResize(data, ArraySize(data) - 1); // Remove null terminator

   ResetLastError();
   int code = WebRequest("POST", url, req_headers, HttpTimeout,
                         data, result, headers);

   if(code < 0 || code == -1)
   {
      int err = GetLastError();
      if(err == 4014)
         Print("[REMORA] ERROR 4014: Add URL to allowed list: ", url,
               " → Tools→Options→Expert Advisors");
      else if(VerboseLog)
         Print("[REMORA] POST failed to ", url, " err=", err);
      return false;
   }

   response = CharArrayToString(result);
   return true;
}

bool HttpGet(const string url, string &response)
{
   char   data[];    // empty for GET
   char   result[];
   string headers;
   string req_headers = "Accept: application/json\r\n";

   ArrayResize(data, 0);
   ResetLastError();
   int code = WebRequest("GET", url, req_headers, HttpTimeout,
                         data, result, headers);

   if(code < 0 || code == -1)
   {
      int err = GetLastError();
      if(err == 4014)
         Print("[REMORA] ERROR 4014: Add URL to allowed list: ", url);
      else if(VerboseLog)
         Print("[REMORA] GET failed err=", err);
      return false;
   }

   response = CharArrayToString(result);
   return true;
}

//── Tick posting ─────────────────────────────────────────────────────

void PostTick()
{
   double bid    = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double ask    = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   double spread = (ask - bid) * MathPow(10, (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS));
   double pnl    = 0.0;

   if(g_in_trade)
   {
      ulong ticket = GetOpenTicket();
      if(ticket > 0 && PositionSelectByTicket(ticket))
         pnl = PositionGetDouble(POSITION_PROFIT);
   }

   string body = StringFormat(
      "{\"type\":\"tick_update\","
      "\"symbol\":\"%s\","
      "\"bid\":%.5f,"
      "\"ask\":%.5f,"
      "\"spread\":%.1f,"
      "\"time\":%d,"
      "\"in_trade\":%s,"
      "\"trade_id\":\"%s\","
      "\"entry_price\":%.5f,"
      "\"current_stop\":%.5f,"
      "\"lots\":%.2f,"
      "\"open_pnl\":%.2f}",
      Symbol(), bid, ask, spread,
      (int)TimeCurrent(),
      g_in_trade ? "true" : "false",
      g_trade_id, g_entry_price, g_current_stop,
      g_lots, pnl
   );

   string resp = "";
   HttpPost(g_tick_url, body, resp);
}

//── Command polling ───────────────────────────────────────────────────

void PollCommands()
{
   string response = "";
   if(!HttpGet(g_poll_url, response))
      return;

   if(StringLen(response) < 10)
      return;

   // Parse "commands" array from JSON
   // Response: {"commands":[{...},{...}],"count":N,"timestamp":"..."}
   int cmds_start = StringFind(response, "\"commands\":[");
   if(cmds_start < 0)
      return;

   // Extract just the array content
   cmds_start += 12; // length of "\"commands\":["
   int cmds_end = StringFind(response, "]", cmds_start);
   if(cmds_end < 0)
      return;

   string array_content = StringSubstr(response, cmds_start,
                                        cmds_end - cmds_start);

   if(StringLen(array_content) < 3)
      return; // empty array

   // Split commands by "},{" boundary
   // Each command is a flat JSON object
   int pos = 0;
   int safety = 0;
   while(pos < StringLen(array_content) && safety < 20)
   {
      safety++;
      // Find start of next object
      int obj_start = StringFind(array_content, "{", pos);
      if(obj_start < 0) break;
      int depth = 1;
      int obj_end = obj_start + 1;
      while(obj_end < StringLen(array_content) && depth > 0)
      {
         ushort c = StringGetCharacter(array_content, obj_end);
         if(c == '{') depth++;
         else if(c == '}') depth--;
         obj_end++;
      }
      string cmd_json = StringSubstr(array_content, obj_start,
                                      obj_end - obj_start);
      if(StringLen(cmd_json) > 3)
         ProcessCommand(cmd_json);
      pos = obj_end;
   }
}

//── Heartbeat ────────────────────────────────────────────────────────

void SendHeartbeat()
{
   string resp = "";
   HttpGet(g_hb_url, resp);
}

//── Command processing ────────────────────────────────────────────────

void ProcessCommand(const string json)
{
   if(VerboseLog) Print("[REMORA] CMD: ", json);

   string cmd_type = JsonExtractString(json, "\"type\"");

   if(cmd_type == "ENTER")             HandleEnter(json);
   else if(cmd_type == "CLOSE_ALL")    HandleCloseAll(JsonExtractString(json, "\"reason\""));
   else if(cmd_type == "CLOSE_PARTIAL")HandleClosePartial(JsonExtractDouble(json, "\"close_fraction\""));
   else if(cmd_type == "TRAIL_STOP")   HandleTrailStop(JsonExtractDouble(json, "\"new_stop\""));
   else if(cmd_type == "MOVE_TO_BE_PLUS") HandleMoveToBePlus(JsonExtractDouble(json, "\"new_stop\""));
}

//── Trade handlers ────────────────────────────────────────────────────

void HandleEnter(const string json)
{
   if(g_in_trade)
   {
      if(VerboseLog) Print("[REMORA] Already in trade — ignoring ENTER");
      return;
   }

   string ea_sym  = JsonExtractString(json, "\"symbol\"");
   string dir_str = JsonExtractString(json, "\"direction\"");
   double lots    = NormLots(JsonExtractDouble(json, "\"lots\""));
   int    sl_pips = (int)JsonExtractDouble(json, "\"initial_sl_pips\"");
   g_trade_id     = JsonExtractString(json, "\"trade_id\"");

   // Accept if symbol matches OR if ea_sym is empty (hub sent to this EA specifically)
   if(StringLen(ea_sym) > 0 && ea_sym != Symbol())
   {
      Print("[REMORA] Symbol mismatch: got ", ea_sym, " expected ", Symbol());
      return;
   }

   int   order_type = (dir_str == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   double point     = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   double digits    = (double)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS);
   double pip       = (digits == 5 || digits == 3) ? point * 10 : point;
   double ask       = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   double bid       = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double entry_p   = (order_type == ORDER_TYPE_BUY) ? ask : bid;
   double sl_price  = (order_type == ORDER_TYPE_BUY)
                        ? entry_p - sl_pips * pip
                        : entry_p + sl_pips * pip;

   // Normalise stop to tick size
   sl_price = NormalizeDouble(sl_price,
                (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS));

   bool ok = (order_type == ORDER_TYPE_BUY)
      ? g_trade.Buy(lots, Symbol(), 0, sl_price, 0, "IMM_P3")
      : g_trade.Sell(lots, Symbol(), 0, sl_price, 0, "IMM_P3");

   if(ok)
   {
      g_in_trade     = true;
      g_entry_price  = entry_p;
      g_direction    = (order_type == ORDER_TYPE_BUY) ? 1 : -1;
      g_lots         = lots;
      g_current_stop = sl_price;

      Print("[REMORA] ENTERED ", dir_str, " ", lots, " lots @ ",
            entry_p, "  SL=", sl_price, "  ID=", g_trade_id);

      PostEvent(StringFormat(
         "{\"type\":\"trade_opened\","
         "\"trade_id\":\"%s\","
         "\"symbol\":\"%s\","
         "\"direction\":\"%s\","
         "\"entry_price\":%.5f,"
         "\"lots\":%.2f,"
         "\"sl_price\":%.5f}",
         g_trade_id, Symbol(), dir_str, entry_p, lots, sl_price
      ));
   }
   else
   {
      int err = GetLastError();
      Print("[REMORA] Order FAILED  err=", err,
            "  retcode=", g_trade.ResultRetcode(),
            "  comment=", g_trade.ResultComment());
      PostEvent(StringFormat(
         "{\"type\":\"order_error\",\"trade_id\":\"%s\",\"error\":%d}",
         g_trade_id, err
      ));
   }
}

void HandleCloseAll(const string reason)
{
   if(!g_in_trade) return;
   double cp = (g_direction == 1)
      ? SymbolInfoDouble(Symbol(), SYMBOL_BID)
      : SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   DoClose(g_lots, cp, reason);
}

void HandleClosePartial(const double fraction)
{
   if(!g_in_trade || fraction <= 0) return;
   double cl = NormLots(g_lots * fraction);
   if(cl >= g_lots - 0.001)
   {
      HandleCloseAll("partial_full");
      return;
   }
   double cp = (g_direction == 1)
      ? SymbolInfoDouble(Symbol(), SYMBOL_BID)
      : SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   bool ok = (g_direction == 1)
      ? g_trade.Sell(cl, Symbol(), 0, 0, 0, "IMM_PARTIAL")
      : g_trade.Buy(cl,  Symbol(), 0, 0, 0, "IMM_PARTIAL");
   if(ok)
   {
      g_lots -= cl;
      if(g_lots < MinLots) g_in_trade = false;
      Print("[REMORA] PARTIAL CLOSE ", cl, " lots remaining=", g_lots);
   }
}

void HandleTrailStop(const double new_stop)
{
   if(!g_in_trade || new_stop <= 0) return;
   // Only move stop in favour direction
   bool move = (g_direction == 1)
      ? (new_stop > g_current_stop + SymbolInfoDouble(Symbol(), SYMBOL_POINT))
      : (new_stop < g_current_stop - SymbolInfoDouble(Symbol(), SYMBOL_POINT));
   if(!move) return;
   ulong ticket = GetOpenTicket();
   if(ticket > 0)
   {
      bool ok = g_trade.PositionModify(ticket, new_stop, 0);
      if(ok)
      {
         g_current_stop = new_stop;
         if(VerboseLog) Print("[REMORA] TRAIL SL→", new_stop);
      }
   }
}

void HandleMoveToBePlus(const double be_stop)
{
   if(!g_in_trade || be_stop <= 0) return;
   ulong ticket = GetOpenTicket();
   if(ticket > 0)
   {
      bool ok = g_trade.PositionModify(ticket, be_stop, 0);
      if(ok)
      {
         g_current_stop = be_stop;
         Print("[REMORA] MOVED TO BE+  SL=", be_stop);
      }
   }
}

//── Internal trade close ──────────────────────────────────────────────

void DoClose(const double lots_to_close, const double close_price,
             const string reason)
{
   bool ok = (g_direction == 1)
      ? g_trade.Sell(lots_to_close, Symbol(), 0, 0, 0, "IMM_CLOSE")
      : g_trade.Buy(lots_to_close,  Symbol(), 0, 0, 0, "IMM_CLOSE");

   if(ok)
   {
      Print("[REMORA] CLOSED  ", lots_to_close, " lots @ ", close_price,
            "  reason=", reason);
      PostEvent(StringFormat(
         "{\"type\":\"trade_closed\","
         "\"trade_id\":\"%s\","
         "\"symbol\":\"%s\","
         "\"close_price\":%.5f,"
         "\"lots\":%.2f,"
         "\"reason\":\"%s\"}",
         g_trade_id, Symbol(), close_price, lots_to_close, reason
      ));
      if(lots_to_close >= g_lots - 0.001)
      {
         g_in_trade = false;
         g_trade_id = "";
         g_lots     = 0;
      }
   }
}

void PostEvent(const string event_json)
{
   string resp = "";
   HttpPost(g_event_url, event_json, resp);
}

//── Utility ───────────────────────────────────────────────────────────

ulong GetOpenTicket()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == Symbol() &&
            PositionGetInteger(POSITION_MAGIC) == 202501)
            return ticket;
      }
   }
   return 0;
}

double NormLots(double lots)
{
   double step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   double minL = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   double maxL = MathMin(SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX), MaxLots);
   lots = MathRound(lots / step) * step;
   return MathMax(minL, MathMin(maxL, NormalizeDouble(lots, 2)));
}

//── Minimal JSON parser (no external lib) ────────────────────────────

string JsonExtractString(const string json, const string key)
{
   int start = StringFind(json, key);
   if(start < 0) return "";
   start = StringFind(json, "\"", start + StringLen(key));
   if(start < 0) return "";
   start++; // skip opening quote
   int end = StringFind(json, "\"", start);
   if(end < 0) return "";
   return StringSubstr(json, start, end - start);
}

double JsonExtractDouble(const string json, const string key)
{
   int start = StringFind(json, key);
   if(start < 0) return 0.0;
   start += StringLen(key);
   // Skip whitespace and colon
   while(start < StringLen(json))
   {
      ushort c = StringGetCharacter(json, start);
      if(c != ':' && c != ' ' && c != '\t') break;
      start++;
   }
   string num = "";
   int safety = 0;
   while(start < StringLen(json) && safety < 30)
   {
      ushort c = StringGetCharacter(json, start);
      if(c == ',' || c == '}' || c == ']' || c == ' ' || c == '\r' || c == '\n')
         break;
      num += ShortToString(c);
      start++;
      safety++;
   }
   if(num == "true")  return 1.0;
   if(num == "false") return 0.0;
   if(num == "null")  return 0.0;
   return StringToDouble(num);
}
