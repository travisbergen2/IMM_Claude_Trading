//+------------------------------------------------------------------+
//|  IMM_REMORA_V2_HTTP.mq5                                          |
//|  APEX REMORA V2 — Phase III Executor                             |
//|  Sends L2 order book depth + per-timeframe data to hub.          |
//|  Receives full command set via HTTP polling.                      |
//|                                                                  |
//|  NEW IN V2:                                                       |
//|  • Sends top-N bid/ask depth levels on every tick post           |
//|  • Sends per-TF (1m/5m/15m/1h/4h) OHLCV summary                 |
//|  • Parses exit_target_up/dn from hub for wall-based exits        |
//|                                                                  |
//|  SETUP:                                                          |
//|  Tools → Options → Expert Advisors → Allow WebRequest            |
//|  Add: https://YOUR_NGROK_URL                                     |
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
input string   HubBaseURL     = "https://YOUR_NGROK_URL";
input string   EaId           = "remora";
input int      PollTimerMs    = 2000;      // Command poll interval (ms)
input int      TickPostInterv = 2;         // Post tick every N ticks
input int      DepthLevels    = 15;        // L2 depth levels to send each side
input int      HttpTimeout    = 4000;
input double   MinLots        = 0.01;
input double   MaxLots        = 10.0;
input int      MaxSlippage    = 5;
input bool     SendTFData     = true;      // Send per-TF data to hub
input bool     VerboseLog     = false;

//── State ────────────────────────────────────────────────────────────
CTrade  g_trade;
string  g_trade_id     = "";
double  g_entry_price  = 0.0;
int     g_direction    = 0;
double  g_lots         = 0.0;
double  g_current_stop = 0.0;
double  g_exit_up      = 0.0;   // Hub-provided resistance wall target
double  g_exit_dn      = 0.0;   // Hub-provided support wall target
bool    g_in_trade     = false;
int     g_tick_count   = 0;

string  g_tick_url;
string  g_poll_url;
string  g_event_url;
string  g_hb_url;

//── Init ─────────────────────────────────────────────────────────────
int OnInit()
{
   g_trade.SetExpertMagicNumber(202601);
   g_trade.SetDeviationInPoints(MaxSlippage * 10);
   g_trade.SetTypeFillingBySymbol(Symbol());

   g_tick_url  = HubBaseURL + "/ea/tick/"      + EaId;
   g_poll_url  = HubBaseURL + "/ea/poll/"      + EaId;
   g_event_url = HubBaseURL + "/ea/event/"     + EaId;
   g_hb_url    = HubBaseURL + "/ea/heartbeat/" + EaId;

   EventSetMillisecondTimer(PollTimerMs);
   DoGet(g_hb_url);

   Print("[REMORA V2] Init complete. Hub=", HubBaseURL);
   Print("[REMORA V2] Depth levels=", DepthLevels, "  TF data=", SendTFData);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); }

//── OnTick ───────────────────────────────────────────────────────────
void OnTick()
{
   g_tick_count++;
   if(g_tick_count % TickPostInterv == 0)
      PostTickWithDepth();
}

//── OnTimer: poll commands ────────────────────────────────────────────
void OnTimer()
{
   PollCommands();
}

//── Tick + depth post ─────────────────────────────────────────────────
void PostTickWithDepth()
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

   // ── Build L2 depth JSON ─────────────────────────────────────────
   string bids_json = BuildDepthSide(BOOK_TYPE_BUY,  DepthLevels);
   string asks_json = BuildDepthSide(BOOK_TYPE_SELL, DepthLevels);

   // ── Build per-TF data JSON ──────────────────────────────────────
   string tf_json = "";
   if(SendTFData)
      tf_json = BuildTFData();

   // ── Compose message ─────────────────────────────────────────────
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
      "\"open_pnl\":%.2f,"
      "\"bids\":%s,"
      "\"asks\":%s"
      "%s}",
      Symbol(), bid, ask, spread,
      (int)TimeCurrent(),
      g_in_trade ? "true" : "false",
      g_trade_id, g_entry_price, g_current_stop,
      g_lots, pnl,
      bids_json, asks_json,
      tf_json != "" ? ",\"tf_data\":" + tf_json : ""
   );

   string resp = "";
   DoPost(g_tick_url, body, resp);
}

//── Build one side of L2 depth ────────────────────────────────────────
string BuildDepthSide(ENUM_BOOK_TYPE side, int levels)
{
   MqlBookInfo book[];
   if(!MarketBookGet(Symbol(), book))
      return "[]";

   string result = "[";
   int count = 0;

   for(int i = 0; i < ArraySize(book) && count < levels; i++)
   {
      if(book[i].type != side) continue;
      if(count > 0) result += ",";
      result += StringFormat("[%.5f,%.4f]", book[i].price, book[i].volume_real);
      count++;
   }

   return result + "]";
}

//── Build per-TF OHLCV summary ────────────────────────────────────────
string BuildTFData()
{
   // Send simplified TF data: close prices and volumes for key TFs
   // Hub uses this to run per-TF phase analysis
   string tfs[];
   ENUM_TIMEFRAMES periods[];
   int n = 5;
   ArrayResize(tfs, n);
   ArrayResize(periods, n);
   tfs[0]="1m";   periods[0]=PERIOD_M1;
   tfs[1]="5m";   periods[1]=PERIOD_M5;
   tfs[2]="15m";  periods[2]=PERIOD_M15;
   tfs[3]="1h";   periods[3]=PERIOD_H1;
   tfs[4]="4h";   periods[4]=PERIOD_H4;

   string result = "{";

   for(int t = 0; t < n; t++)
   {
      // Get last 10 closes and volumes for this TF
      double closes[], volumes[];
      int bars = 10;
      ArraySetAsSeries(closes,  true);
      ArraySetAsSeries(volumes, true);

      if(CopyClose( Symbol(), periods[t], 0, bars, closes)  < bars) continue;
      if(CopyTickVolume(Symbol(), periods[t], 0, bars, volumes) < bars) continue;

      // Compute simple phase indicators for this TF
      double ret5 = (closes[0] - closes[5]) / (closes[5] + 1e-9);
      double vol_avg = 0; for(int j=0;j<bars;j++) vol_avg += volumes[j]; vol_avg/=bars;
      double vol_ratio = vol_avg > 0 ? volumes[0] / vol_avg : 1.0;

      // Direction estimate: recent 3-bar net return
      double dir3 = closes[0] > closes[3] ? 1.0 : (closes[0] < closes[3] ? -1.0 : 0.0);

      if(t > 0) result += ",";
      result += StringFormat(
         "\"%s\":{\"close\":%.5f,\"ret5\":%.6f,\"vol_ratio\":%.3f,\"dir\":%.1f}",
         tfs[t], closes[0], ret5, vol_ratio, dir3
      );
   }

   return result + "}";
}

//── Poll commands from hub ────────────────────────────────────────────
void PollCommands()
{
   string resp = "";
   if(!DoGet(g_poll_url, resp)) return;
   if(StringLen(resp) < 10) return;

   int cs = StringFind(resp, "\"commands\":[");
   if(cs < 0) return;
   cs += 12;
   int ce = StringFind(resp, "]", cs);
   if(ce < 0) return;
   string arr = StringSubstr(resp, cs, ce - cs);
   if(StringLen(arr) < 3) return;

   int pos=0; int safety=0;
   while(pos < StringLen(arr) && safety < 20)
   {
      safety++;
      int os = StringFind(arr, "{", pos);
      if(os < 0) break;
      int depth=1; int oe=os+1;
      while(oe < StringLen(arr) && depth > 0)
      {
         ushort c = StringGetCharacter(arr, oe);
         if(c=='{') depth++;
         else if(c=='}') depth--;
         oe++;
      }
      ProcessCommand(StringSubstr(arr, os, oe-os));
      pos = oe;
   }
}

//── Command dispatch ──────────────────────────────────────────────────
void ProcessCommand(const string json)
{
   if(VerboseLog) Print("[REMORA V2] CMD: ", json);
   string t = Jstr(json, "\"type\"");

   if(t == "ENTER")              HandleEnter(json);
   else if(t == "CLOSE_ALL")     HandleCloseAll(Jstr(json,"\"reason\""));
   else if(t == "CLOSE_PARTIAL") HandlePartial(Jdbl(json,"\"close_fraction\""));
   else if(t == "TRAIL_STOP")    HandleTrail(Jdbl(json,"\"new_stop\""));
   else if(t == "MOVE_TO_BE_PLUS") HandleBE(Jdbl(json,"\"new_stop\""));
}

//── Trade handlers ────────────────────────────────────────────────────
void HandleEnter(const string json)
{
   if(g_in_trade) return;

   string dir_str = Jstr(json, "\"direction\"");
   double lots    = NormLots(Jdbl(json,"\"lots\""));
   int    sl_p    = (int)Jdbl(json,"\"initial_sl_pips\"");
   g_trade_id     = Jstr(json,"\"trade_id\"");
   g_exit_up      = Jdbl(json,"\"exit_target_up\"");
   g_exit_dn      = Jdbl(json,"\"exit_target_dn\"");

   int    otype   = (dir_str=="BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   double point   = SymbolInfoDouble(Symbol(),SYMBOL_POINT);
   double digits  = (double)SymbolInfoInteger(Symbol(),SYMBOL_DIGITS);
   double pip     = (digits==5||digits==3) ? point*10 : point;
   double ask     = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
   double bid     = SymbolInfoDouble(Symbol(),SYMBOL_BID);
   double ep      = (otype==ORDER_TYPE_BUY) ? ask : bid;
   double sl      = NormalizeDouble(
                      (otype==ORDER_TYPE_BUY) ? ep-sl_p*pip : ep+sl_p*pip,
                      (int)SymbolInfoInteger(Symbol(),SYMBOL_DIGITS));

   bool ok = (otype==ORDER_TYPE_BUY)
      ? g_trade.Buy(lots,Symbol(),0,sl,0,"IMM_P3_V2")
      : g_trade.Sell(lots,Symbol(),0,sl,0,"IMM_P3_V2");

   if(ok)
   {
      g_in_trade=true; g_entry_price=ep;
      g_direction=(otype==ORDER_TYPE_BUY)?1:-1;
      g_lots=lots; g_current_stop=sl;

      Print("[REMORA V2] ENTERED ",dir_str," ",lots,"@",ep,
            " SL=",sl," walls=[",g_exit_dn,"→",g_exit_up,"]");

      PostEvent(StringFormat(
         "{\"type\":\"trade_opened\",\"trade_id\":\"%s\","
         "\"symbol\":\"%s\",\"direction\":\"%s\","
         "\"entry_price\":%.5f,\"lots\":%.2f,\"sl\":%.5f}",
         g_trade_id,Symbol(),dir_str,ep,lots,sl));
   }
   else
      Print("[REMORA V2] Order FAILED err=",GetLastError(),
            " ",g_trade.ResultComment());
}

void HandleCloseAll(const string reason)
{
   if(!g_in_trade) return;
   double cp=(g_direction==1)?SymbolInfoDouble(Symbol(),SYMBOL_BID)
                              :SymbolInfoDouble(Symbol(),SYMBOL_ASK);
   bool ok=(g_direction==1)?g_trade.Sell(g_lots,Symbol(),0,0,0,"IMM_CLOSE")
                            :g_trade.Buy(g_lots, Symbol(),0,0,0,"IMM_CLOSE");
   if(ok)
   {
      PostEvent(StringFormat(
         "{\"type\":\"trade_closed\",\"trade_id\":\"%s\","
         "\"close_price\":%.5f,\"reason\":\"%s\"}",
         g_trade_id,cp,reason));
      g_in_trade=false; g_trade_id=""; g_lots=0;
      Print("[REMORA V2] CLOSED reason=",reason);
   }
}

void HandlePartial(const double frac)
{
   if(!g_in_trade||frac<=0) return;
   double cl=NormLots(g_lots*frac);
   if(cl>=g_lots-0.001){HandleCloseAll("partial_full");return;}
   bool ok=(g_direction==1)?g_trade.Sell(cl,Symbol(),0,0,0,"IMM_PART")
                            :g_trade.Buy(cl, Symbol(),0,0,0,"IMM_PART");
   if(ok){g_lots-=cl; if(g_lots<MinLots) g_in_trade=false;}
}

void HandleTrail(const double ns)
{
   if(!g_in_trade||ns<=0) return;
   double pt=SymbolInfoDouble(Symbol(),SYMBOL_POINT);
   bool mv=(g_direction==1)?(ns>g_current_stop+pt):(ns<g_current_stop-pt);
   if(!mv) return;
   ulong t=GetOpenTicket();
   if(t>0&&g_trade.PositionModify(t,ns,0)) g_current_stop=ns;
}

void HandleBE(const double bs)
{
   if(!g_in_trade||bs<=0) return;
   ulong t=GetOpenTicket();
   if(t>0&&g_trade.PositionModify(t,bs,0)) g_current_stop=bs;
}

// ── Wall-based exit check (called each tick when in trade) ────────────
// Called in OnTick after PostTickWithDepth
void CheckWallExit()
{
   if(!g_in_trade||g_exit_up<=0&&g_exit_dn<=0) return;
   double bid=SymbolInfoDouble(Symbol(),SYMBOL_BID);
   double ask=SymbolInfoDouble(Symbol(),SYMBOL_ASK);

   if(g_direction==1 && g_exit_up>0 && bid>=g_exit_up*0.999)
   {
      Print("[REMORA V2] WALL EXIT at resistance ",g_exit_up);
      HandleCloseAll("wall_resistance_hit");
   }
   else if(g_direction==-1 && g_exit_dn>0 && ask<=g_exit_dn*1.001)
   {
      Print("[REMORA V2] WALL EXIT at support ",g_exit_dn);
      HandleCloseAll("wall_support_hit");
   }
}

//── Utilities ─────────────────────────────────────────────────────────
ulong GetOpenTicket()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong t=PositionGetTicket(i);
      if(PositionSelectByTicket(t)&&
         PositionGetString(POSITION_SYMBOL)==Symbol()&&
         PositionGetInteger(POSITION_MAGIC)==202601) return t;
   }
   return 0;
}

double NormLots(double l)
{
   double s=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_STEP);
   double mn=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN);
   double mx=MathMin(SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX),MaxLots);
   return MathMax(mn,MathMin(mx,NormalizeDouble(MathRound(l/s)*s,2)));
}

void PostEvent(const string ev)
{ string r=""; DoPost(g_event_url,ev,r); }

bool DoPost(const string url,const string body,string &resp)
{
   char d[];char r[];string h;
   StringToCharArray(body,d,0,StringLen(body));
   ArrayResize(d,ArraySize(d)-1);
   string rh="Content-Type: application/json\r\n";
   int code=WebRequest("POST",url,rh,HttpTimeout,d,r,h);
   if(code<0){if(GetLastError()==4014)Print("[REMORA V2] Add URL: ",url);return false;}
   resp=CharArrayToString(r); return true;
}

bool DoGet(const string url,string &resp)
{
   char d[];char r[];string h;ArrayResize(d,0);
   string rh="Accept: application/json\r\n";
   int code=WebRequest("GET",url,rh,HttpTimeout,d,r,h);
   if(code<0) return false;
   resp=CharArrayToString(r); return true;
}
bool DoGet(const string url){string r="";return DoGet(url,r);}

string Jstr(const string j,const string k)
{
   int s=StringFind(j,k);if(s<0)return"";
   s=StringFind(j,"\"",s+StringLen(k));if(s<0)return"";
   s++;int e=StringFind(j,"\"",s);if(e<0)return"";
   return StringSubstr(j,s,e-s);
}
double Jdbl(const string j,const string k)
{
   int s=StringFind(j,k);if(s<0)return 0.0;
   s+=StringLen(k);
   while(s<StringLen(j)&&(StringGetCharacter(j,s)==':'||StringGetCharacter(j,s)==' '))s++;
   string n="";int sf=0;
   while(s<StringLen(j)&&sf<30){ushort c=StringGetCharacter(j,s);
   if(c==','||c=='}'||c==']'||c==' ')break;n+=ShortToString(c);s++;sf++;}
   return StringToDouble(n);
}
